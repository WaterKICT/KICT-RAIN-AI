# -*- coding: utf-8 -*-
"""
기상청 HSR 500m QPE(합성장) → AI 예측강우 생산 (추론)   [전체 모델 .h5 전용 ]

처리 흐름
  기상청 HSR 합성장 4프레임(t-30/-20/-10/0, 10분 간격)
    → 784×784 @500m 클립 (학습과 동일 창: CLIP_R0=910, CLIP_C0=932) → mm/h → /255
    → 전체 모델 .h5 → +10 ~ +60분 예측강우 6프레임 (mm/h)

입력 : RDR_CMP_HSP_PUB_YYYYMMDDHHMM.bin.gz
       (1024B 헤더 + int16 LE 2305×2881, 값×0.01=mm/h, 결측 -30000 / 무강우 -25000,
        원자료 row=0 은 남쪽 → 판독 시 상하반전)
       경로: <root>/YYYYMM/DD/RDR_CMP_HSP_PUB_YYYYMMDDHHMM.bin.gz
출력 : (1) .npy  (784, 784, 6)  mm/h
       (2) CSV   선행시간별 784×784 격자 (헤더 없는 격자 형식)

사전학습 가중치 파일 다운로드 : https://drive.google.com/file/d/1oek-SdhtGS2LGwpbz1JmrIfGYf2ZnbAR/view?usp=drive_link
실행 예
  # HSR 원본에서 직접
  python predict_KMA_QPE_500M_h5.py --model model-best_rec_60min_final.h5 \
         --root E:/FOR_KAIST/02_RADAR_QPE_QPF/01_QPE --t0 202607080400 --outdir pred_out
  # 전처리 A(v1) 시퀀스 .npy(10,784,784)로 추론 + 관측 대비 간단 검증
  python predict_KMA_QPE_500M_h5.py --model model-best_rec_60min_final.h5 \
         --seq-npy hsr_seq_500m/seq_202106302000.npy --outdir pred_out
"""
import os
import sys
import gzip
import zlib
from datetime import datetime, timedelta

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# 1. HSR 격자·클립·시퀀스 사양  (학습 전처리와 동일 값)
# ══════════════════════════════════════════════════════════════════════════════
NX, NY = 2305, 2881              # EW(col), NS(row)
HEADER_BYTES = 1024
MISS_NODATA, MISS_NORAIN = -30000, -25000
HSR_SCALE = 0.01                 # 값 × 0.01 = mm/h

CLIP_H, CLIP_W = 784, 784        # 예측모델 입력용 데이터 레이더 격자영역 (500m)
CLIP_R0, CLIP_C0 = 910, 932      # 북쪽-위 좌표 기준 클립 시작 (2026-07-08 04:00 정합)
ROW_NORTH_FIRST = False          # 원자료 row=0 은 남쪽

N_IN, N_LEAD, STEP_MIN = 4, 6, 10
RAIN_SCALE = 255.0               # mm/h → /255 정규화 (학습과 동일)
RAIN_THRESH_MMH = 0.1

H, W = CLIP_H, CLIP_W
LEAD_MIN = [STEP_MIN * (i + 1) for i in range(N_LEAD)]   # [10,...,60]
IN_OFFSETS = [-STEP_MIN * k for k in range(N_IN - 1, -1, -1)]   # [-30,-20,-10,0]

BAD_FILE_ERRORS = (EOFError, OSError, ValueError, zlib.error)


# ── HSR 판독 ──────────────────────────────────────────────────────────────────
def _raw_int16(path):
    """.bin(.gz) → (NY, NX) int16 뷰(복사 없음)."""
    with open(path, "rb") as f:
        raw = f.read()
    if path.endswith(".gz"):
        raw = gzip.decompress(raw)
    exp = HEADER_BYTES + NX * NY * 2
    if len(raw) != exp:
        raise ValueError("파일 크기 불일치: %s (%d, 기대 %d)"
                         % (os.path.basename(path), len(raw), exp))
    return np.frombuffer(raw[HEADER_BYTES:], dtype="<i2").reshape(NY, NX)


def _clip_slice_int16(a, r0, c0, h, w):
    """북쪽-위 좌표(r0,c0,h,w)에 해당하는 int16 부분배열(상하반전 반영)."""
    if c0 < 0 or c0 + w > NX:
        raise ValueError("클립 열 범위 초과: c0=%d w=%d (NX=%d)" % (c0, w, NX))
    if r0 < 0 or r0 + h > NY:
        raise ValueError("클립 행 범위 초과: r0=%d h=%d (NY=%d)" % (r0, h, NY))
    if ROW_NORTH_FIRST:
        return a[r0:r0 + h, c0:c0 + w]
    s0, s1 = NY - (r0 + h), NY - r0
    return a[s0:s1, c0:c0 + w][::-1, :]


def read_clip(path, r0=CLIP_R0, c0=CLIP_C0, h=CLIP_H, w=CLIP_W):
    """HSR 파일 → 클립영역 (h,w) float32 mm/h. 결측 → NaN."""
    sub = _clip_slice_int16(_raw_int16(path), r0, c0, h, w)
    out = sub.astype(np.float32) * HSR_SCALE
    out[(sub == MISS_NODATA) | (sub == MISS_NORAIN)] = np.nan
    out[out < 0] = np.nan
    return out


def read_clip_zero(path, r0=CLIP_R0, c0=CLIP_C0, h=CLIP_H, w=CLIP_W):
    """read_clip 과 동일하되 결측·음수를 0(무강우)으로. 모델 입력용(학습과 동일)."""
    a = read_clip(path, r0, c0, h, w)
    a[np.isnan(a)] = 0.0
    return a


def rain_area_fraction(frame):
    """강우 셀(≥RAIN_THRESH_MMH) 면적비 [0~1]."""
    return float(np.count_nonzero(frame >= RAIN_THRESH_MMH)) / frame.size


def hsr_path(root, dt):
    fname = "RDR_CMP_HSP_PUB_%s.bin.gz" % dt.strftime("%Y%m%d%H%M")
    return os.path.join(root, dt.strftime("%Y%m"), dt.strftime("%d"), fname)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 입력텐서 (기상청 HSR → 784×784×4)
# ══════════════════════════════════════════════════════════════════════════════
def input_times(t0):
    """t0 기준 입력 4프레임 시각 [t-30, t-20, t-10, t0]."""
    return [t0 + timedelta(minutes=o) for o in IN_OFFSETS]


def input_paths(root, t0):
    return [hsr_path(root, t) for t in input_times(t0)]


def build_input_tensor(root, t0, r0=CLIP_R0, c0=CLIP_C0):
    """HSR 4프레임 → (H, W, 4) float32, /255 정규화. 결측·무강우는 0.

    파일 누락·손상 시 어느 시각이 문제인지 명시해 중단한다.
    """
    paths = input_paths(root, t0)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("입력 HSR 파일 누락 %d개:\n  %s"
                                % (len(missing), "\n  ".join(missing)))
    frames = []
    for p in paths:
        try:
            frames.append(read_clip_zero(p, r0=r0, c0=c0, h=H, w=W))
        except BAD_FILE_ERRORS as ex:
            raise RuntimeError("HSR 판독 실패 %s : %s: %s"
                               % (os.path.basename(p), type(ex).__name__, ex))
    seq = np.stack(frames, axis=-1)                            # (H,W,4) mm/h
    return (seq / RAIN_SCALE).astype(np.float32)


def load_seq_npy(path):
    """전처리 A(v1) 시퀀스 .npy (10,H,W) mm/h → 입력 (H,W,4) 정규화, 타깃 (H,W,6) mm/h."""
    seq = np.load(path).astype(np.float32)
    if seq.ndim != 3 or seq.shape[0] < N_IN:
        raise ValueError("시퀀스 .npy 형상 오류 %s : %s (기대 (%d+,%d,%d))"
                         % (path, seq.shape, N_IN, H, W))
    x = np.transpose(seq[:N_IN], (1, 2, 0)) / RAIN_SCALE
    y = None
    if seq.shape[0] >= N_IN + N_LEAD:
        y = np.transpose(seq[N_IN:N_IN + N_LEAD], (1, 2, 0))
    return x.astype(np.float32), y


# ══════════════════════════════════════════════════════════════════════════════
# 3. 모델 적재 (전체 모델 .h5 만; 구조 정의 없음)
# ══════════════════════════════════════════════════════════════════════════════
def _check_full_model(path):
    """model.save() 전체 모델인지 확인. 가중치 전용 파일이면 안내 후 중단."""
    if not os.path.exists(path):
        raise FileNotFoundError("모델 파일 없음: %s" % path)
    try:
        import h5py
    except ImportError:
        return
    with h5py.File(path, "r") as f:
        if "model_config" not in f.attrs:
            raise RuntimeError(
                "%s 는 가중치 전용 파일(save_weights_only)이라 구조 정보가 없습니다.\n"
                "학습 환경에서 model.save('…_final.h5') 로 저장한 전체 모델 파일을 지정하세요."
                % os.path.basename(path))


def load_model(path):
    """전체 모델 .h5 → tf.keras 모델. 컴파일 정보(옵티마이저·손실)는 추론에 불필요하므로 생략."""
    _check_full_model(path)
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    try:
        model = tf.keras.models.load_model(path, compile=False)
    except TypeError:
        # Keras 3: 람다/슬라이싱 레이어가 든 레거시 .h5 는 safe_mode=False 가 필요할 수 있음
        model = tf.keras.models.load_model(path, compile=False, safe_mode=False)
    print("모델 적재:", path)

    ishape = tuple(model.input_shape[1:])
    oshape = tuple(model.output_shape[1:])
    if ishape != (H, W, N_IN):
        raise RuntimeError("모델 입력 %s ≠ 기대 %s" % (ishape, (H, W, N_IN)))
    if oshape[-1] != N_LEAD:
        raise RuntimeError("모델 출력 채널 %d ≠ 기대 %d" % (oshape[-1], N_LEAD))
    return model


def predict(model, x):
    """x (H,W,4) 정규화 입력 → (H,W,n_lead) mm/h 예측. 음수는 0으로."""
    y = model.predict(x[np.newaxis, ...], verbose=0)[0]
    y = np.asarray(y, dtype=np.float32) * RAIN_SCALE
    y[y < 0] = 0.0
    return y


# ══════════════════════════════════════════════════════════════════════════════
# 4. 저장/요약
# ══════════════════════════════════════════════════════════════════════════════
def save_csv(arr2d, path, fmt="%.2f"):
    """784×784 격자를 헤더 없는 CSV 로 저장."""
    np.savetxt(path, arr2d, delimiter=",", fmt=fmt)


def save_outputs(pred, t0_str, outdir, save_csv_files=True, prefix="PredRain_KMA"):
    """예측 (H,W,6) → .npy 1개 + 선행시간별 CSV 6개."""
    os.makedirs(outdir, exist_ok=True)
    npy = os.path.join(outdir, "%s_%s.npy" % (prefix, t0_str))
    np.save(npy, pred.astype(np.float32))
    made = [npy]
    if save_csv_files:
        for i, lead in enumerate(LEAD_MIN):
            p = os.path.join(outdir, "%s_%s_p%02d.csv" % (prefix, t0_str, lead))
            save_csv(pred[:, :, i], p)
            made.append(p)
    return made


def summarize_input(x, t0):
    """입력 4프레임의 강우면적·최대(입력이 비어 있지 않은지 확인용)."""
    print("  입력    시각            강우면적(>=0.1)   최대 mm/h")
    for i, t in enumerate(input_times(t0)):
        f = x[:, :, i] * RAIN_SCALE
        print("  t%+3d분  %s   %10.2f%%   %9.2f"
              % (IN_OFFSETS[i], t.strftime("%Y-%m-%d %H:%M"),
                 100 * rain_area_fraction(f), f.max()))


def summarize(pred, obs=None):
    """선행시간별 강우면적·최대·평균 요약. 관측(obs)이 있으면 MAE·CSI(≥1mm/h) 도 출력."""
    hdr = "  선행    강우면적(>=0.1)   최대 mm/h   평균 mm/h"
    if obs is not None:
        hdr += "   MAE mm/h   CSI(>=1)"
    print(hdr)
    for i, lead in enumerate(LEAD_MIN):
        f = pred[:, :, i]
        line = "  +%2d분   %10.2f%%   %9.2f   %9.4f" % (
            lead, 100 * (f >= RAIN_THRESH_MMH).mean(), f.max(), f.mean())
        if obs is not None:
            o = obs[:, :, i]
            hit = np.count_nonzero((f >= 1) & (o >= 1))
            fa = np.count_nonzero((f >= 1) & (o < 1))
            ms = np.count_nonzero((f < 1) & (o >= 1))
            csi = hit / float(hit + fa + ms) if (hit + fa + ms) else float("nan")
            line += "   %8.4f   %8.3f" % (np.abs(f - o).mean(), csi)
        print(line)


# ══════════════════════════════════════════════════════════════════════════════
# 5. 메인
# ══════════════════════════════════════════════════════════════════════════════
def _preset_gpu(argv):
    """TF import 전에 CUDA_VISIBLE_DEVICES 를 설정(--gpu)."""
    for i, a in enumerate(argv):
        if a == "--gpu" and i + 1 < len(argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = argv[i + 1]
        elif a.startswith("--gpu="):
            os.environ["CUDA_VISIBLE_DEVICES"] = a.split("=", 1)[1]


def main(argv=None):
    import argparse

    argv = sys.argv[1:] if argv is None else argv
    _preset_gpu(argv)
    ap = argparse.ArgumentParser(description="기상청 HSR 500m QPE → AI 예측강우 (추론, 전체 모델 .h5 전용)")
    ap.add_argument("--model", required=True, help="model.save() 로 저장한 전체 모델 .h5")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", help="HSR 루트 (…/YYYYMM/DD/RDR_CMP_HSP_PUB_*.bin.gz)")
    src.add_argument("--seq-npy", help="전처리 A(v1) 시퀀스 .npy (10,784,784) 로 추론")
    ap.add_argument("--t0", help="기준시각 YYYYMMDDHHMM (가장 최근 관측). --root 시 필수")
    ap.add_argument("--r0", type=int, default=CLIP_R0, help="클립 시작 행(북쪽 위)")
    ap.add_argument("--c0", type=int, default=CLIP_C0, help="클립 시작 열")
    ap.add_argument("--outdir", default="pred_out")
    ap.add_argument("--prefix", default="PredRain_KMA", help="출력 파일 접두어")
    ap.add_argument("--gpu", default="", help="사용할 GPU 번호(비우면 기본)")
    ap.add_argument("--no-csv", action="store_true", help="CSV 저장 생략(.npy 만)")
    ap.add_argument("--save-input", action="store_true",
                    help="입력 4프레임(mm/h)도 .npy 로 저장")
    a = ap.parse_args(argv)

    obs = None
    if a.root:
        if not a.t0:
            ap.error("--root 사용 시 --t0 가 필요합니다")
        t0 = datetime.strptime(a.t0, "%Y%m%d%H%M")
        t0_str = a.t0
        print("클립창 r0=%d c0=%d %dx%d | 입력 %d프레임 → 출력 %d프레임(+%d~+%d분)"
              % (a.r0, a.c0, H, W, N_IN, N_LEAD, LEAD_MIN[0], LEAD_MIN[-1]))
        x = build_input_tensor(a.root, t0, r0=a.r0, c0=a.c0)
    else:
        base = os.path.splitext(os.path.basename(a.seq_npy))[0]
        t0_str = a.t0 or base.replace("seq_", "")
        t0 = datetime.strptime(t0_str, "%Y%m%d%H%M")
        x, obs = load_seq_npy(a.seq_npy)
        print("시퀀스 .npy 적재: %s (관측 타깃 %s)"
              % (a.seq_npy, "있음" if obs is not None else "없음"))
    summarize_input(x, t0)
    print("입력 텐서:", x.shape, "값범위 %.5f~%.5f" % (x.min(), x.max()))

    model = load_model(a.model)
    pred = predict(model, x)
    print("예측:", pred.shape, "mm/h")
    summarize(pred, obs)

    made = save_outputs(pred, t0_str, a.outdir, save_csv_files=not a.no_csv, prefix=a.prefix)
    if a.save_input:
        p = os.path.join(a.outdir, "%s_%s_input.npy" % (a.prefix, t0_str))
        np.save(p, (x * RAIN_SCALE).astype(np.float32))
        made.append(p)
    for p in made:
        print("  저장:", p)
    return pred


if __name__ == "__main__":
    main()
