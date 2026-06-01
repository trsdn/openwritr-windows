import os, sys, time, traceback
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

mdl = os.environ["MDL"]
log = open(os.path.join(mdl, "qnn_test.log"), "w", encoding="utf-8", buffering=1)

def p(*args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()

p("ort:", ort.__version__, "qnn:", qnn_ep.__version__)
ep = "QNNExecutionProvider"
ort.register_execution_provider_library(ep, qnn_ep.get_library_path())
devs = [d for d in ort.get_ep_devices() if d.ep_name == ep]
p("QNN EP devices found:", len(devs))
for d in devs:
    p("  ", d.device.vendor, d.device.type, d.device.device_id)

so = ort.SessionOptions()
so.add_provider_for_devices(devs, {
    "backend_path": qnn_ep.get_qnn_htp_path(),
    "htp_performance_mode": "burst",
    "enable_htp_fp16_precision": "1",
})
p("loading encoder INT8 on QNN HTP ...")
t0 = time.time()
try:
    enc = ort.InferenceSession(os.path.join(mdl, "encoder-model.int8.onnx"), sess_options=so)
    p(f"  loaded in {time.time()-t0:.1f}s, providers:", enc.get_providers())
except Exception as e:
    p("ENCODER LOAD FAILED:", type(e).__name__)
    p(traceback.format_exc())
log.close()
