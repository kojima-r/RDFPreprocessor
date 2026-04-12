python - <<'PY'
import torch
print("torch       :", torch.__version__)

try:
        import torchrec
            print("torchrec    :", torchrec.__version__)
except Exception as e:
        print("torchrec    : import failed", e)

        try:
                import fbgemm_gpu
                    print("fbgemm_gpu  :", getattr(fbgemm_gpu, "__version__", "unknown"))
        except Exception as e:
                print("fbgemm_gpu  : import failed", e)
                PY
