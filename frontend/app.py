"""Streamlit control plane (PRD F-05). EDA: bounded PCM only. Full jobs: CLI subprocess (NF-05).

Retro-futurist 「录音窗口」UI: orange / white / amber, CRT scanlines — cosmetic only; logic unchanged.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Repo root on path for ``from src...`` (same as CI / README).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.axes import Axes
from numpy.typing import NDArray
import streamlit as st

# —— Visual tokens: 橙 / 白 / 琥珀 + CRT ——
_REC_BG = "#141008"
_REC_PANEL = "#1c140c"
_REC_DEEP = "#0f0c08"
_ORANGE = "#ea580c"
_ORANGE_HI = "#f97316"
_AMBER = "#fbbf24"
_AMBER_HI = "#fcd34d"
_TEXT = "#fff7ed"
_TEXT_MUTED = "#fde8d0"
_WARN = "#fb923c"

_REC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Audiowide&family=Noto+Sans+SC:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
  font-family: 'Noto Sans SC', 'Segoe UI', system-ui, sans-serif;
  font-size: 15px;
  line-height: 1.6;
}
.stApp {
  position: relative !important;
  background: radial-gradient(ellipse 100% 90% at 50% 0%, #2d1f14 0%, #141008 42%, #0f0c08 100%) !important;
}
/* CRT vignette */
.stApp::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 9998;
  background: radial-gradient(ellipse 70% 65% at 50% 45%, transparent 30%, rgba(0,0,0,0.45) 100%);
}
/* CRT scanlines */
.stApp::after {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 9999;
  opacity: 0.09;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.45) 2px,
    rgba(0,0,0,0.45) 3px
  );
}
[data-testid="stHeader"] {
  background: rgba(20,16,8,0.92) !important;
  border-bottom: 1px solid rgba(234,88,12,0.35);
}
[data-testid="stToolbar"] { visibility: hidden; height: 0; }
section[data-testid="stSidebar"] {
  background: linear-gradient(175deg, #241a10 0%, #1a140c 50%, #141008 100%) !important;
  border-right: 1px solid rgba(251,191,36,0.28);
  box-shadow: inset -1px 0 0 rgba(234,88,12,0.12);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
  font-family: 'Audiowide', 'Noto Sans SC', sans-serif !important;
  letter-spacing: 0.06em !important;
  color: #fcd34d !important;
  text-transform: none !important;
  font-size: 0.95rem !important;
  font-weight: 600 !important;
}
h1, h2 {
  font-family: 'Audiowide', 'Noto Sans SC', sans-serif !important;
  letter-spacing: 0.06em;
  color: #fef3c7 !important;
}
h1 { text-shadow: 0 0 20px rgba(251,191,36,0.25); }
.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
  background: rgba(28,20,12,0.75);
  padding: 8px;
  border-radius: 4px;
  border: 1px solid rgba(234,88,12,0.35);
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(180deg, rgba(234,88,12,0.35) 0%, rgba(120,50,10,0.4) 100%) !important;
  color: #fff7ed !important;
  border: 1px solid rgba(251,191,36,0.55) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1px solid rgba(234,88,12,0.4) !important;
  border-radius: 4px;
  background: linear-gradient(145deg, rgba(36,26,16,0.96) 0%, rgba(20,16,8,0.98) 100%);
  box-shadow: 0 0 28px rgba(234,88,12,0.08), inset 0 1px 0 rgba(255,247,237,0.06);
}
.stButton > button {
  font-family: 'Noto Sans SC', sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
  border-radius: 4px !important;
  border: 1px solid rgba(251,191,36,0.55) !important;
  background: linear-gradient(180deg, #c2410c 0%, #9a3412 100%) !important;
  color: #fffbeb !important;
  box-shadow: 0 0 18px rgba(234,88,12,0.25);
}
.stButton > button:hover {
  border-color: #fef3c7 !important;
  box-shadow: 0 0 22px rgba(251,191,36,0.35);
}
label, span[data-baseweb="tag"] { color: #fde8d0 !important; }
[data-testid="stMarkdownContainer"] a { color: #fdba74 !important; }
.stCodeBlock, pre {
  border-left: 3px solid #ea580c !important;
  background: #0f0c08 !important;
  color: #fff7ed !important;
}
[data-testid="stCaption"] { color: #fde8d0 !important; }
"""

_REC_BANNER = """
<div style="
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:14px;
  padding:16px 20px;margin-bottom:16px;
  background:linear-gradient(95deg,#1c140c 0%,#2d1810 45%,#141008 100%);
  border:1px solid rgba(234,88,12,0.45);
  border-radius:4px;
  box-shadow:0 0 36px rgba(234,88,12,0.12), inset 0 1px 0 rgba(255,247,237,0.08);
">
  <div style="font-family:'Audiowide',sans-serif;font-weight:400;font-size:1.2rem;letter-spacing:0.08em;color:#fef3c7;">
    HANKEL · STEREO · PURIFY
  </div>
  <div style="font-size:0.8rem;letter-spacing:0.12em;color:rgba(253,232,176,0.75);">
    录音窗口 · MSSA · PRD F-05 · NF-05
  </div>
</div>
<div style="font-size:0.88rem;color:rgba(255,247,237,0.88);margin:-4px 0 18px 0;line-height:1.65;max-width:52rem;">
  复古未来主义控制平面 · EDA 仅读取预览时长内 PCM · 全量任务仅通过 CLI 子进程（内存隔离）
</div>
"""


def _inject_recording_crt_theme() -> None:
    st.markdown(f"<style>{_REC_CSS}</style>", unsafe_allow_html=True)


def _repo_pythonpath() -> str:
    return str(_REPO_ROOT)


def _build_full_batch_cli_cmd(
    *,
    inp: Path,
    outp: Path,
    window_length: int,
    max_mem_mb: int,
    mode_fixed_rank: bool,
    rank_or_energy: float | int,
    frame_size: int | None,
    hop: int | None,
) -> list[str]:
    """Assemble ``python -m src.cli …`` argv for the full-batch tab (testable, no shell)."""
    cmd: list[str] = [
        sys.executable,
        "-m",
        "src.cli",
        str(inp),
        str(outp),
        "-L",
        str(int(window_length)),
        "--max-memory-mb",
        str(int(max_mem_mb)),
    ]
    if mode_fixed_rank:
        cmd.extend(["-k", str(int(rank_or_energy))])
    else:
        cmd.extend(["--energy-fraction", str(rank_or_energy)])
    if frame_size is not None:
        cmd.extend(["--frame-size", str(int(frame_size))])
    if hop is not None:
        cmd.extend(["--hop", str(int(hop))])
    return cmd


def _load_preview_pcm(path: Path, max_seconds: float) -> tuple[NDArray[np.float64], int]:
    """Read at most ``max_seconds`` of audio (never full-file load for huge files)."""
    with sf.SoundFile(path) as f:
        sr = int(f.samplerate)
        ch = int(f.channels)
        if ch != 2:
            raise ValueError("Stereo (2 ch) required for MSSA.")
        max_frames = int(max_seconds * sr)
        n = min(max_frames, int(f.frames))
        if n <= 0:
            raise ValueError("No frames to read.")
        data = f.read(frames=n, dtype="float64", always_2d=True)
    return np.asarray(data, dtype=np.float64), sr


def _run_eda_denoise(
    pcm: NDArray[np.float64],
    samplerate: int,
    *,
    window_length: int,
    use_energy: bool,
    rank_or_energy: float,
    frame_size: int | None,
    hop_size: int | None,
    max_mem_mb: int,
) -> NDArray[np.float64]:
    """In-process denoise for **preview segment only** (small arrays)."""
    from src.facade.purifier import MSSAPurifierBuilder

    b = (
        MSSAPurifierBuilder()
        .set_window_length(window_length)
        .set_max_working_memory_bytes(max_mem_mb * 1024 * 1024)
    )
    if use_energy:
        b = b.set_energy_fraction(float(rank_or_energy))
    else:
        b = b.set_truncation_rank(int(rank_or_energy))
    if frame_size is not None:
        b = b.set_frame_size(frame_size)
    if hop_size is not None:
        b = b.set_hop_size(hop_size)
    purifier = b.build()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t_in:
        in_path = Path(t_in.name)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t_out:
        out_path = Path(t_out.name)
    try:
        sf.write(in_path, pcm, samplerate, subtype="PCM_24")
        purifier.process_file(str(in_path), str(out_path))
        out, _ = sf.read(out_path, dtype="float64", always_2d=True)
        return np.asarray(out, dtype=np.float64)
    finally:
        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


def _plot_spectra(ax: Axes, y: NDArray[np.float64], sr: int, title_zh: str) -> None:
    mono = np.mean(y, axis=1)
    ax.specgram(mono, Fs=sr, NFFT=1024, noverlap=512, cmap="inferno")
    ax.set_title(title_zh, color=_AMBER_HI, fontsize=12, fontweight="600")
    ax.set_ylabel("频率 (Hz)", color=_TEXT_MUTED, fontsize=10)
    ax.set_xlabel("时间 (s)", color=_TEXT_MUTED, fontsize=10)
    ax.tick_params(colors="#a8a29e", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(_ORANGE)
        spine.set_alpha(0.55)


def main() -> None:
    st.set_page_config(
        page_title="HSP · 录音窗口",
        layout="wide",
        page_icon="🎙️",
    )
    _inject_recording_crt_theme()
    st.markdown(_REC_BANNER, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("#### 参数条 · REC")
        st.caption("与 CLI 同源")
        window_length = st.number_input("-L / window-length", min_value=1, value=256, step=1)
        mode = st.radio(
            "截断模式",
            ["固定秩 -k", "能量 --energy-fraction"],
            index=0,
        )
        if mode == "固定秩 -k":
            rank = st.number_input("rank k", min_value=1, value=64, step=1)
            use_energy = False
            rank_or_energy = float(rank)
        else:
            ef = st.slider("energy-fraction", min_value=0.01, max_value=1.0, value=0.95)
            use_energy = True
            rank_or_energy = float(ef)
        frame_size = st.number_input(
            "--frame-size（空=默认）",
            min_value=0,
            value=0,
            step=1,
            help="0 表示不传该参数",
        )
        hop = st.number_input(
            "--hop（空=默认）",
            min_value=0,
            value=0,
            step=1,
        )
        max_mem_mb = st.number_input("--max-memory-mb", min_value=1, value=1500, step=1)
        cli_timeout_sec = st.number_input(
            "全量 CLI 子进程超时（秒）",
            min_value=0,
            value=0,
            step=60,
            help="0 表示不限制；过大文件处理可设为 86400（24h）等。",
        )

    fs_none = int(frame_size) if int(frame_size) > 0 else None
    hop_none = int(hop) if int(hop) > 0 else None

    tab_eda, tab_full = st.tabs(["录音预览 · 频谱", "全量导出 · CLI"])

    with tab_eda:
        with st.container(border=True):
            st.markdown(
                f"##### 录音窗口 · 仅读取前 **N** 秒 PCM  "
                f"<span style='color:{_WARN};font-size:0.8rem'>· 非整文件载入</span>",
                unsafe_allow_html=True,
            )
            prev_sec = st.slider("预览时长（秒）", min_value=0.5, max_value=30.0, value=5.0, step=0.5)
            src = st.radio("信号源", ["本地路径", "上传小文件"], horizontal=True)
            path: Path | None = None
            if src == "本地路径":
                raw = st.text_input("音频文件路径（绝对或相对）", value="")
                if raw.strip():
                    path = Path(raw).expanduser().resolve()
            else:
                up = st.file_uploader("立体声 WAV / FLAC（建议 < 50MB）", type=["wav", "flac"])
                if up is not None:
                    suf = Path(up.name).suffix or ".wav"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
                        tmp.write(up.getbuffer())
                        st.session_state["eda_preview_path"] = tmp.name
                path_str = st.session_state.get("eda_preview_path")
                if path_str:
                    path = Path(path_str)

            if st.button("生成降噪频谱对比", type="primary") and path is not None:
                if not path.is_file():
                    st.error("文件不存在或路径无效。")
                else:
                    try:
                        pcm_in, sr = _load_preview_pcm(path, prev_sec)
                        if pcm_in.shape[0] < 32:
                            st.warning("样本过短，请增大预览时长。")
                        else:
                            with st.spinner("降噪处理中…"):
                                pcm_out = _run_eda_denoise(
                                    pcm_in,
                                    sr,
                                    window_length=int(window_length),
                                    use_energy=use_energy,
                                    rank_or_energy=rank_or_energy,
                                    frame_size=fs_none,
                                    hop_size=hop_none,
                                    max_mem_mb=int(max_mem_mb),
                                )
                            st.success("左：输入 · 右：输出（并排频谱对比）")
                            fig, (ax0, ax1) = plt.subplots(
                                1,
                                2,
                                figsize=(14, 5.2),
                                facecolor=_REC_BG,
                            )
                            fig.patch.set_facecolor(_REC_BG)
                            fig.suptitle(
                                "频谱对比（预览段）",
                                color=_TEXT,
                                fontsize=14,
                                fontweight="600",
                                y=1.02,
                            )
                            for ax in (ax0, ax1):
                                ax.set_facecolor(_REC_PANEL)
                            _plot_spectra(ax0, pcm_in, sr, "降噪前（预览）")
                            _plot_spectra(ax1, pcm_out, sr, "降噪后（预览）")
                            plt.tight_layout(rect=(0, 0, 1, 0.92))
                            st.pyplot(fig, clear_figure=True)
                            plt.close(fig)
                            st.caption(
                                "提示：两图同一时间轴尺度；inferno 配色为能量强度。"
                            )
                    except Exception as exc:
                        st.exception(exc)

    with tab_full:
        with st.container(border=True):
            st.markdown(
                f"##### 批处理窗口 · CLI 子进程  "
                f"<span style='color:{_ORANGE_HI};font-size:0.8rem'>· NF-05 内存隔离</span>",
                unsafe_allow_html=True,
            )
            in_full = st.text_input("输入 · 绝对路径", key="full_in")
            out_full = st.text_input("输出 · 绝对路径", key="full_out")
            if st.button("启动 CLI 子进程", type="primary", key="btn_full"):
                if not in_full.strip() or not out_full.strip():
                    st.error("请填写输入与输出路径。")
                else:
                    inp = Path(in_full).expanduser().resolve()
                    outp = Path(out_full).expanduser().resolve()
                    cmd = _build_full_batch_cli_cmd(
                        inp=inp,
                        outp=outp,
                        window_length=int(window_length),
                        max_mem_mb=int(max_mem_mb),
                        mode_fixed_rank=(mode == "固定秩 -k"),
                        rank_or_energy=rank_or_energy,
                        frame_size=(
                            int(fs_none) if fs_none is not None else None
                        ),
                        hop=int(hop_none) if hop_none is not None else None,
                    )
                    env = {**os.environ, "PYTHONPATH": _repo_pythonpath()}
                    st.code(" ".join(cmd), language="bash")
                    timeout_s: float | None = (
                        None if int(cli_timeout_sec) <= 0 else float(cli_timeout_sec)
                    )
                    try:
                        proc = subprocess.run(
                            cmd,
                            cwd=str(_REPO_ROOT),
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=timeout_s,
                        )
                        st.text_area("stdout", proc.stdout or "(empty)", height=160)
                        st.text_area("stderr", proc.stderr or "(empty)", height=160)
                        st.info(f"退出码: {proc.returncode}")
                    except subprocess.TimeoutExpired as exc:
                        st.error(
                            "子进程超时：请增大「全量 CLI 子进程超时」、检查磁盘/NFS，"
                            "或对大文件直接使用命令行 `python -m src.cli`。"
                        )
                        if exc.output:
                            st.text_area("stdout (partial)", exc.output, height=120)
                        if exc.stderr:
                            st.text_area("stderr (partial)", exc.stderr, height=120)
                    except Exception as exc:
                        st.exception(exc)


if __name__ == "__main__":
    main()
