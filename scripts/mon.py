import subprocess, os, time
def r(c):
    try:
        return subprocess.check_output(c, shell=True, stderr=subprocess.DEVNULL).decode()
    except Exception as e:
        return "ERR:" + str(e)
print("=== " + time.strftime("%H:%M:%S") + " ===")
print("snapshot.sh:", ("RUNNING" if "scripts/snapshot.sh" in (r("ps -eo cmd 2>/dev/null") or "") else "not running"))
build = r("ls -td /home/trinity/data/prod_ready_ganjuur/2026*_build 2>/dev/null | head -1").strip()
s1 = "/home/trinity/data/prod_ready_ganjuur/snapshot1"
if build:
    print("build size:", r("du -sh " + build + " 2>/dev/null").strip())
    print("build frame#:", r("find " + build + "/data/ganjuur_crops/db_frames -type f 2>/dev/null | wc -l").strip())
if os.path.isdir(s1):
    print("*** snapshot1 READY *** size=", r("du -sh " + s1 + " 2>/dev/null").strip())
    print("snapshot1 frames:", r("find " + s1 + "/data/ganjuur_crops/db_frames -type f 2>/dev/null | wc -l").strip())
    print("snapshot1 qdrant snaps:", r("find " + s1 + "/qdrant_snapshot -name '*.snapshot' 2>/dev/null | wc -l").strip())
    print("snapshot1 full_storage:", r("du -sh " + s1 + "/qdrant_snapshot/full_qdrant_storage 2>/dev/null").strip() or "(none)")
    print("snapshot1 config:", r("ls " + s1 + "/config/ 2>/dev/null").strip() or "(none)")
    print("snapshot1 extras:", r("ls " + s1 + "/*.py " + s1 + "/asset/ " + s1 + "/*.md " + s1 + "/*.json " + s1 + "/*.db " + s1 + "/*.html 2>/dev/null").strip() or "(none)")
print()
log = build + "/logs/snapshot.log" if build else "/tmp/snap_v4.log"
lt = r("tail -18 " + log) if os.path.exists(log) else r("tail -18 /tmp/snap_v4.log 2>/dev/null") or "(no log yet)"
print("--- LOG TAIL ---")
print(lt)
