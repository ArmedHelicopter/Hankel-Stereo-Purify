"""Stable user-facing strings for ``AudioIOError`` (DRY across facade and I/O)."""


def audio_io_failed_pair(input_path: str, output_path: str) -> str:
    return f"Audio I/O failed (input={input_path!r}, output={output_path!r})"


def failed_to_open_audio_file(path: str) -> str:
    return f"Failed to open audio file: {path}"


def failed_to_read_audio_stream(path: str) -> str:
    return f"Failed to read audio stream: {path}"


def input_file_does_not_exist(input_path: str) -> str:
    return f"Input file does not exist: {input_path}"


def input_path_not_a_file(input_path: str) -> str:
    return f"Input path is not a file: {input_path}"


def unable_to_create_output_directory(parent: str) -> str:
    return f"Unable to create output directory: {parent}"


def empty_or_invalid_audio_length() -> str:
    return "Empty or invalid audio length."


def ola_memmap_allocation_failed() -> str:
    return "Failed to allocate temp buffers for OLA (memmap)."


def pcm_stream_ended_before_ola_complete() -> str:
    return "PCM stream ended before enough samples were available for OLA."


def block_size_must_be_positive() -> str:
    return "block_size must be positive."
