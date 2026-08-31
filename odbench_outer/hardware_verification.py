"""Trusted host-side verification for physical deployment targets."""

from __future__ import annotations

import fcntl
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from .hardware import HardwareTarget


PROFILE_DURATION = re.compile(
    r"duration\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*ms by sample"
    r"\s*\(([0-9.]+)/([0-9.]+)/([0-9.]+)\)"
)
PROFILE_THROUGHPUT = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s+inf/s")
PROFILE_NODES = re.compile(r"n_nodes\s*:\s*([0-9]+)")
EPOCH_TOTAL = re.compile(r"Total number of epochs\s+([0-9]+)", re.IGNORECASE)
EPOCH_SOFTWARE = re.compile(r"pure software \(SW\) epochs\s+([0-9]+)", re.IGNORECASE)
EPOCH_HYBRID = re.compile(r"hybrid epochs.*?\s+([0-9]+)\s*$", re.IGNORECASE | re.MULTILINE)
EPOCH_HARDWARE = re.compile(
    r"pure hardware \(HW or EC\) epochs\s+([0-9]+)", re.IGNORECASE
)
MEMORY_TOTAL = re.compile(
    r"^Total:\s+([0-9.]+)\s*(B|kB|MB|GB).*?"
    r"weights:\s+([0-9.]+)\s*(B|kB|MB|GB).*?"
    r"activations:\s+([0-9.]+)\s*(B|kB|MB|GB)",
    re.IGNORECASE | re.MULTILINE,
)
MEMORY_POOL = re.compile(
    r"file postfix=(\S+)\s+name=\S+\s+offset=(0x[0-9a-fA-F]+)"
)
MAX_RETURNED_LOG_CHARS = 32_000


class HardwareVerificationError(RuntimeError):
    pass


class VerificationStageError(HardwareVerificationError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tail(value: str, limit: int = MAX_RETURNED_LOG_CHARS) -> str:
    return value[-limit:]


def parse_checker_profile(output: str) -> dict[str, Any]:
    duration = PROFILE_DURATION.search(output)
    if duration is None:
        raise HardwareVerificationError("ST checker output did not contain a duration")
    throughput = PROFILE_THROUGHPUT.search(output)
    nodes = PROFILE_NODES.search(output)
    return {
        "duration_seconds": round(float(duration.group(1)) / 1000.0, 12),
        "duration_ms": {
            "mean": float(duration.group(1)),
            "min": float(duration.group(2)),
            "max": float(duration.group(3)),
            "stddev": float(duration.group(4)),
        },
        "inferences_per_second": (
            float(throughput.group(1)) if throughput is not None else None
        ),
        "nodes": int(nodes.group(1)) if nodes is not None else None,
    }


def _memory_bytes(value: str, unit: str) -> int:
    factors = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3}
    return round(float(value) * factors[unit.lower()])


def parse_generation_report(output: str) -> dict[str, Any]:
    """Extract accelerator epoch coverage and memory from an ST compiler report."""

    matches = {
        "total_epochs": EPOCH_TOTAL.search(output),
        "software_epochs": EPOCH_SOFTWARE.search(output),
        "hybrid_epochs": EPOCH_HYBRID.search(output),
        "hardware_epochs": EPOCH_HARDWARE.search(output),
    }
    values = {
        name: int(match.group(1)) if match is not None else None
        for name, match in matches.items()
    }
    total = values["total_epochs"]
    software = values["software_epochs"]
    hybrid = values["hybrid_epochs"]
    hardware = values["hardware_epochs"]
    warnings: list[str] = []
    accelerator_epochs = None
    accelerator_percent = None
    if None not in (total, software, hybrid, hardware):
        assert total is not None and software is not None
        assert hybrid is not None and hardware is not None
        if software + hybrid + hardware != total:
            warnings.append(
                "compiler epoch counts are inconsistent; inspect generation_report"
            )
        if total > 0:
            accelerator_epochs = hardware + hybrid
            accelerator_percent = round(100.0 * accelerator_epochs / total, 3)
            if accelerator_epochs == 0:
                warnings.append(
                    "the compiler mapped 0% of epochs to the accelerator; expect CPU-like timing"
                )
            elif software > 0:
                warnings.append(
                    f"{software} pure software epoch(s) remain and may dominate latency"
                )
    else:
        warnings.append(
            "accelerator epoch counts were not found in the compiler report"
        )
    result: dict[str, Any] = {
        **values,
        "accelerated_epochs": accelerator_epochs,
        "accelerator_epoch_percent": accelerator_percent,
        "warnings": warnings,
    }
    memory = MEMORY_TOTAL.search(output)
    if memory is not None:
        result["memory"] = {
            "total_bytes": _memory_bytes(memory.group(1), memory.group(2)),
            "weights_bytes": _memory_bytes(memory.group(3), memory.group(4)),
            "activations_bytes": _memory_bytes(memory.group(5), memory.group(6)),
        }
    else:
        result["memory"] = None
    return result


def diagnose_profile_failure(
    error: str,
    profile_output: str,
) -> dict[str, Any] | None:
    """Classify a device timeout that happens after model execution starts."""

    if "STM32 - read timeout" not in error or "Running c-model" not in profile_output:
        return None
    return {
        "kind": "model_execution_timeout",
        "phase": "inference",
        "board_connection_succeeded": True,
        "model_discovery_succeeded": True,
        "inference_started": True,
        "retry_unchanged_model": False,
        "summary": (
            "Hardware setup succeeded: the board connected, the model was discovered, "
            "and inference started. The device then stopped responding before reporting "
            "an operator result. This was a model-execution timeout, not an initial "
            "board connection or flashing failure."
        ),
        "next_action": (
            "Do not retry the unchanged model. Try a smaller input or activation "
            "footprint, fewer or narrower layers, or different fully accelerated "
            "operators; then run analyze_for_hw and verify_on_hw on the changed graph "
            "before spending more training budget."
        ),
    }


def external_memory_images(network_c: Path) -> list[tuple[Path, str | None]]:
    """Resolve generated external-memory images and raw-image load addresses."""

    output = network_c.parent
    images: list[tuple[Path, str | None]] = [
        (path, None) for path in sorted(output.glob("network_atonbuf.xSPI*.hex"))
    ]
    raw_images = sorted(output.glob("network_atonbuf.xSPI*.raw"))
    if not raw_images:
        return images
    source = network_c.read_text(encoding="utf-8", errors="replace")
    addresses = {match.group(1): match.group(2) for match in MEMORY_POOL.finditer(source)}
    for path in raw_images:
        # network_atonbuf.xSPI2.raw -> xSPI2
        postfix = path.name.removeprefix("network_atonbuf.").removesuffix(".raw")
        address = addresses.get(postfix)
        if address is None:
            raise HardwareVerificationError(
                f"cannot resolve load address for generated memory image {path.name}"
            )
        images.append((path, address))
    return sorted(images, key=lambda item: item[0].name)


def cubeprogrammer_download_image(image: Path, directory: Path) -> Path:
    """Copy raw compiler output to a filename CubeProgrammer accepts."""

    if image.suffix.lower() != ".raw":
        return image
    download_image = directory / f"{image.name}.bin"
    shutil.copy2(image, download_image)
    return download_image


def external_memory_image_records(
    images: list[tuple[Path, str | None]],
) -> list[dict[str, Any]]:
    return [
        {
            "name": image.name,
            "bytes": image.stat().st_size,
            "address": address,
        }
        for image, address in images
    ]


def validate_external_memory_image_sizes(
    images: list[tuple[Path, str | None]], *, max_image_bytes: int
) -> None:
    for image, _ in images:
        image_bytes = image.stat().st_size
        if image_bytes > max_image_bytes:
            raise HardwareVerificationError(
                f"generated external-memory image {image.name} is {image_bytes} bytes; "
                f"target limit is {max_image_bytes} bytes"
            )


class NucleoN657Verifier:
    """Generate, flash, and profile one ONNX model on a NUCLEO-N657X0-Q."""

    def __init__(
        self,
        target: HardwareTarget,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if target.kind != "stm32n6" or target.board != "NUCLEO-N657X0-Q":
            raise ValueError(f"unsupported hardware target: {target.target_id}")
        self.target = target
        self.environment = dict(os.environ if environment is None else environment)

    def _stedgeai_root(self) -> Path:
        return Path(
            self.environment.get("ODBENCH_STEDGEAI_ROOT", "/Applications/ST/STEdgeAI/4.0")
        )

    def _stedgeai_binary(self) -> Path:
        override = self.environment.get("ODBENCH_STEDGEAI_BIN")
        if override:
            return Path(override)
        system = platform.system().lower()
        machine = platform.machine().lower()
        platform_dir = "macarm" if system == "darwin" and machine == "arm64" else system
        return self._stedgeai_root() / "Utilities" / platform_dir / "stedgeai"

    def _cubeide_path(self) -> Path:
        return Path(
            self.environment.get(
                "ODBENCH_CUBEIDE_PATH",
                "/Applications/STM32CubeIDE.app/Contents/Eclipse",
            )
        )

    def _programmer(self, cubeide: Path) -> Path:
        override = self.environment.get("ODBENCH_STM32_PROGRAMMER_CLI")
        if override:
            return Path(override)
        standalone = Path(
            "/Applications/STMicroelectronics/STM32Cube/STM32CubeProgrammer/"
            "STM32CubeProgrammer.app/Contents/Resources/bin/STM32_Programmer_CLI"
        )
        bundled = sorted(
            cubeide.glob(
                "plugins/com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.*/"
                "tools/bin/STM32_Programmer_CLI"
            )
        )
        candidates = [standalone, *bundled]
        return next((path for path in candidates if path.is_file()), standalone)

    def _serial_port(self) -> str:
        override = self.environment.get("ODBENCH_NUCLEO_SERIAL_PORT")
        if override:
            if not Path(override).exists():
                raise HardwareVerificationError(
                    f"configured Nucleo serial port does not exist: {override}"
                )
            return override
        candidates: list[str] = []
        for pattern in ("/dev/cu.usbmodem*", "/dev/ttyACM*"):
            candidates.extend(glob.glob(pattern))
        candidates = sorted(set(candidates))
        if len(candidates) != 1:
            raise HardwareVerificationError(
                "expected exactly one Nucleo serial port; set "
                "ODBENCH_NUCLEO_SERIAL_PORT explicitly "
                f"(found {candidates or 'none'})"
            )
        return candidates[0]

    @staticmethod
    def _required_file(path: Path, description: str) -> Path:
        if not path.is_file():
            raise HardwareVerificationError(f"{description} is missing: {path}")
        return path

    def _run(
        self,
        stage: str,
        command: list[str],
        *,
        cwd: Path,
        logs: Path,
        timeout: float,
        environment: Mapping[str, str] | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        started = time.monotonic()
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                env=dict(environment or self.environment),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout if isinstance(error.stdout, str) else ""
            stderr = error.stderr if isinstance(error.stderr, str) else ""
            (logs / f"{stage}.stdout.log").write_text(stdout, encoding="utf-8")
            (logs / f"{stage}.stderr.log").write_text(stderr, encoding="utf-8")
            raise VerificationStageError(stage, f"{stage} timed out after {timeout:g}s") from error
        elapsed = time.monotonic() - started
        (logs / f"{stage}.stdout.log").write_text(result.stdout, encoding="utf-8")
        (logs / f"{stage}.stderr.log").write_text(result.stderr, encoding="utf-8")
        (logs / f"{stage}.command.json").write_text(
            json.dumps(
                {
                    "argv": command,
                    "elapsed_seconds": elapsed,
                    "returncode": result.returncode,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if check and result.returncode != 0:
            detail = _tail((result.stderr or result.stdout).strip(), 8_000)
            raise VerificationStageError(
                stage,
                detail or f"{stage} exited with status {result.returncode}",
            )
        return result

    def _generate(
        self,
        model: Path,
        *,
        work: Path,
        logs: Path,
        report_directory: Path,
    ) -> tuple[Path, str, dict[str, Any]]:
        output = work / "st_ai_output"
        workspace = work / "st_ai_workspace"
        edge_binary = self._required_file(
            self._stedgeai_binary(), "ST Edge AI executable"
        )
        generated = self._run(
            "generate",
            [
                str(edge_binary),
                "generate",
                "--model",
                str(model),
                "--target",
                "stm32n6",
                "--st-neural-art",
                "--output",
                str(output),
                "--workspace",
                str(workspace),
            ],
            cwd=work,
            logs=logs,
            timeout=900,
        )
        network_c = self._required_file(output / "network.c", "generated network.c")
        generation_output = generated.stdout
        for name in ("network_generate_report.txt", "network_analyze_report.txt"):
            candidate = output / name
            if candidate.is_file():
                generation_output = candidate.read_text(
                    encoding="utf-8", errors="replace"
                )
                shutil.copy2(candidate, report_directory / name)
                break
        return network_c, generation_output, parse_generation_report(generation_output)

    def analyze(self, model: Path, *, report_directory: Path) -> dict[str, Any]:
        """Compile one model and report accelerator mapping without touching the board."""

        report_directory.mkdir(parents=True, exist_ok=False)
        logs = report_directory / "logs"
        logs.mkdir()
        started_at = time.time()
        report: dict[str, Any] = {
            "schema_version": 1,
            "type": "hardware_analysis",
            "target": self.target.public_metadata(),
            "model": {
                "format": "onnx",
                "bytes": model.stat().st_size if model.is_file() else None,
                "sha256": _sha256(model) if model.is_file() else None,
            },
            "compiled": False,
            "stage": "environment",
            "started_at_unix": started_at,
        }
        generation_output = ""
        try:
            self._required_file(model, "ONNX model")
            if model.suffix.lower() != ".onnx":
                raise HardwareVerificationError("hardware analysis requires an .onnx file")
            self._required_file(self._stedgeai_binary(), "ST Edge AI executable")
            lock_path = Path(tempfile.gettempdir()) / f"odbench-{self.target.target_id}.lock"
            with lock_path.open("a+", encoding="utf-8") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                with tempfile.TemporaryDirectory(
                    prefix=f"odbench-analyze-{self.target.target_id}-"
                ) as temporary:
                    report["stage"] = "generate"
                    network_c, generation_output, mapping = self._generate(
                        model,
                        work=Path(temporary),
                        logs=logs,
                        report_directory=report_directory,
                    )
                    report["accelerator_mapping"] = mapping
                    report["stage"] = "external_memory"
                    external_images = external_memory_images(network_c)
                    report["external_memory_images"] = external_memory_image_records(
                        external_images
                    )
                    validate_external_memory_image_sizes(
                        external_images,
                        max_image_bytes=self.target.max_external_image_bytes,
                    )
                    report["compiled"] = True
                    report["stage"] = "complete"
        except VerificationStageError as error:
            report["stage"] = error.stage
            report["error"] = str(error)
        except (HardwareVerificationError, OSError) as error:
            report["error"] = str(error)
        report["elapsed_seconds"] = time.time() - started_at
        report["generation_report"] = _tail(generation_output)
        (report_directory / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report

    def verify(
        self,
        model: Path,
        *,
        allowed_runtime_seconds: float,
        acceptance_mode: str,
        report_directory: Path,
    ) -> dict[str, Any]:
        report_directory.mkdir(parents=True, exist_ok=False)
        logs = report_directory / "logs"
        logs.mkdir()
        started_at = time.time()
        runtime_tolerance = (
            self.target.submission_tolerance_fraction
            if acceptance_mode == "submission"
            else 0.0
        )
        target_runtime_seconds = round(
            allowed_runtime_seconds / (1.0 + runtime_tolerance), 12
        )
        report: dict[str, Any] = {
            "schema_version": 1,
            "type": "hardware_verification",
            "target": self.target.public_metadata(
                runtime_seconds=target_runtime_seconds
            ),
            "acceptance_mode": acceptance_mode,
            "target_runtime_seconds": target_runtime_seconds,
            "allowed_runtime_seconds": allowed_runtime_seconds,
            "samples": self.target.benchmark_samples,
            "model": {
                "format": "onnx",
                "bytes": model.stat().st_size if model.is_file() else None,
                "sha256": _sha256(model) if model.is_file() else None,
            },
            "passed": False,
            "stage": "environment",
            "started_at_unix": started_at,
        }
        checker_output = ""
        generation_output = ""
        try:
            self._required_file(model, "ONNX model")
            if model.suffix.lower() != ".onnx":
                raise HardwareVerificationError("hardware verification requires an .onnx file")
            edge_root = self._stedgeai_root()
            self._required_file(self._stedgeai_binary(), "ST Edge AI executable")
            cubeide = self._cubeide_path()
            if not cubeide.is_dir():
                raise HardwareVerificationError(f"STM32CubeIDE path is missing: {cubeide}")
            programmer = self._required_file(
                self._programmer(cubeide), "STM32CubeProgrammer CLI"
            )
            external_loader = self._required_file(
                programmer.parent
                / "ExternalLoader"
                / "MX25UM51245G_STM32N6570-NUCLEO.stldr",
                "Nucleo external-flash loader",
            )
            loader = self._required_file(
                edge_root / "scripts" / "N6_scripts" / "n6_loader.py",
                "ST N6 loader",
            )
            loader_wrapper = self._required_file(
                Path(__file__).with_name("n6_loader_wrapper.py"),
                "od-benchmark N6 loader wrapper",
            )
            checker = self._required_file(
                edge_root / "scripts" / "ai_runner" / "examples" / "checker.py",
                "ST AI Runner checker",
            )

            lock_path = Path(tempfile.gettempdir()) / f"odbench-{self.target.target_id}.lock"
            with lock_path.open("a+", encoding="utf-8") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                with tempfile.TemporaryDirectory(
                    prefix=f"odbench-{self.target.target_id}-"
                ) as temporary:
                    work = Path(temporary)
                    config = work / "n6_config.json"
                    config.write_text(
                        json.dumps(
                            {
                                "compiler_type": "gcc",
                                "cubeide_path": str(cubeide),
                            },
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )

                    report["stage"] = "generate"
                    network_c, generation_output, mapping = self._generate(
                        model,
                        work=work,
                        logs=logs,
                        report_directory=report_directory,
                    )
                    report["accelerator_mapping"] = mapping
                    output = network_c.parent

                    report["stage"] = "external_memory"
                    external_images = external_memory_images(network_c)
                    report["external_memory_images"] = external_memory_image_records(
                        external_images
                    )
                    validate_external_memory_image_sizes(
                        external_images,
                        max_image_bytes=self.target.max_external_image_bytes,
                    )

                    stlink_serial = self.environment.get("ODBENCH_STLINK_SERIAL")
                    report["stage"] = "reset"
                    reset = self._run(
                        "reset",
                        [
                            str(programmer),
                            "-q",
                            "-c",
                            "port=SWD",
                            *( [f"sn={stlink_serial}"] if stlink_serial else [] ),
                            "mode=powerdown",
                            "freq=200",
                            "ap=1",
                        ],
                        cwd=work,
                        logs=logs,
                        timeout=60,
                        check=False,
                    )
                    if reset.returncode != 0:
                        report["reset_warning"] = _tail(
                            (reset.stderr or reset.stdout).strip(), 8_000
                        )
                    connection = [
                        "port=SWD",
                        "mode=HOTPLUG",
                        # ST's N6 loader requires SWD below 1000 kHz for external flash.
                        "freq=200",
                        "ap=1",
                    ]
                    if stlink_serial:
                        connection.insert(1, f"sn={stlink_serial}")
                    for index, (image, address) in enumerate(external_images, start=1):
                        stage = f"flash_external_{index}"
                        report["stage"] = stage
                        download_image = cubeprogrammer_download_image(image, work)
                        download = ["--download", str(download_image)]
                        if address is not None:
                            download.append(address)
                        self._run(
                            stage,
                            [
                                str(programmer),
                                "-q",
                                "-c",
                                *connection,
                                "--extload",
                                str(external_loader),
                                *download,
                                "--verify",
                            ],
                            cwd=work,
                            logs=logs,
                            timeout=self.target.external_flash_timeout_seconds,
                        )
                    report["stage"] = "build_and_load"
                    loader_command = [
                        sys.executable,
                        str(loader_wrapper),
                        str(loader),
                        "--config",
                        str(config),
                        "--network-file",
                        str(network_c),
                        "--build-config",
                        "N6-Nucleo",
                        "--clean",
                        "--skip-flash",
                    ]
                    if stlink_serial:
                        loader_command.extend(["--serial-number", stlink_serial])
                    self._run(
                        "build_and_load",
                        loader_command,
                        cwd=report_directory,
                        logs=logs,
                        timeout=900,
                    )

                    report["stage"] = "profile"
                    serial_port = self._serial_port()
                    checker_environment = dict(self.environment)
                    ai_runner = edge_root / "scripts" / "ai_runner"
                    existing_pythonpath = checker_environment.get("PYTHONPATH")
                    checker_environment["PYTHONPATH"] = (
                        str(ai_runner)
                        if not existing_pythonpath
                        else f"{ai_runner}{os.pathsep}{existing_pythonpath}"
                    )
                    checker_environment.setdefault(
                        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python"
                    )
                    checked = self._run(
                        "profile",
                        [
                            sys.executable,
                            str(checker),
                            "--desc",
                            f"serial:{serial_port}:921600",
                            "--perf-only",
                            "--batch",
                            str(self.target.benchmark_samples),
                            "--verbosity",
                            "1",
                        ],
                        cwd=work,
                        logs=logs,
                        timeout=180,
                        environment=checker_environment,
                    )
                    checker_output = checked.stdout
                    profile = parse_checker_profile(checker_output)
                    report.update(profile)
                    report["passed"] = (
                        profile["duration_seconds"] <= allowed_runtime_seconds
                    )
                    report["stage"] = "complete"
        except VerificationStageError as error:
            report["stage"] = error.stage
            raw_error = str(error)
            if error.stage == "profile":
                try:
                    checker_output = (logs / "profile.stdout.log").read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError:
                    pass
                failure = diagnose_profile_failure(raw_error, checker_output)
                if failure is not None:
                    report["failure"] = failure
                    report["raw_error"] = raw_error
                    report["error"] = failure["summary"]
                else:
                    report["error"] = raw_error
            else:
                report["error"] = raw_error
        except (HardwareVerificationError, OSError) as error:
            report["error"] = str(error)
        report["elapsed_seconds"] = time.time() - started_at
        report["generation_report"] = _tail(generation_output)
        report["profile_report"] = _tail(checker_output)
        (report_directory / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
