# Find a working way to install Nsight Systems on the Kaggle T4 image and prove
# that profile -> stats -> sqlite export works. Output is captured verbatim.
log("nsys probe start")
report = {}
cmds = [
    ("os", "cat /etc/os-release | head -2; dpkg --print-architecture; id -u"),
    ("apt-sources", "ls /etc/apt/sources.list.d/ 2>&1; grep -rh 'nvidia' /etc/apt/sources.list.d/ 2>/dev/null | head -5"),
    ("apt-update", "DEBIAN_FRONTEND=noninteractive apt-get update 2>&1 | tail -8; echo rc=${PIPESTATUS[0]}"),
    ("apt-search", "apt-cache search nsight-systems 2>&1 | head -10; apt-cache policy cuda-nsight-systems-12-8 2>&1 | head -5"),
    ("install-meta", "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends cuda-nsight-systems-12-8 2>&1 | tail -6"),
    ("find-nsys-1", "ls -d /opt/nvidia/nsight-systems*/*/bin /usr/local/cuda*/bin 2>/dev/null; find / -name nsys -type f 2>/dev/null | head -5"),
]
for name, cmd in cmds:
    rc, out, err = sh(f"bash -c '{cmd}'" if "PIPESTATUS" in cmd else cmd, timeout=1200)
    print(f"$ [{name}] {cmd}\n{out}\n{err}\n", flush=True)
    report[name] = {"rc": rc, "out": out[-3000:], "err": err[-800:]}

rc, nsys_path, _ = sh("find / -name nsys -type f 2>/dev/null | head -1")
if not nsys_path:
    log("meta package failed; trying keyring + explicit package")
    for name, cmd in [
        ("keyring", "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/k.deb && dpkg -i /tmp/k.deb 2>&1 | tail -2 && DEBIAN_FRONTEND=noninteractive apt-get update 2>&1 | tail -3"),
        ("apt-search-2", "apt-cache search nsight-systems 2>&1 | head -10"),
        ("install-2", "PKG=$(apt-cache search nsight-systems | awk '{print $1}' | grep -E '^nsight-systems-20' | sort -V | tail -1); echo PKG=$PKG; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends ${PKG:-cuda-nsight-systems-12-8} 2>&1 | tail -6"),
        ("find-nsys-2", "find / -name nsys -type f 2>/dev/null | head -5"),
    ]:
        rc, out, err = sh(cmd, timeout=1200)
        print(f"$ [{name}] {cmd}\n{out}\n{err}\n", flush=True)
        report[name] = {"rc": rc, "out": out[-3000:], "err": err[-800:]}
    rc, nsys_path, _ = sh("find / -name nsys -type f 2>/dev/null | head -1")

report["nsys_path"] = nsys_path
if nsys_path:
    nsys_bin = os.path.dirname(nsys_path)
    os.environ["PATH"] = nsys_bin + ":" + os.environ["PATH"]
    smoke = "import cupy as cp; x = cp.arange(1<<24, dtype=cp.float32); [ (x*2.0).sum() for _ in range(5) ]; cp.cuda.Device().synchronize()"
    for name, cmd in [
        ("version", "nsys --version"),
        ("profile", f"cd /kaggle/working && nsys profile --trace=cuda,nvtx --stats=false -o {OUT}/raw/probe_nsys python -c \"{smoke}\" 2>&1 | tail -4"),
        ("stats", f"nsys stats --report cuda_gpu_kern_sum --format csv --output {OUT}/raw/probe_kern_sum {OUT}/raw/probe_nsys.nsys-rep 2>&1 | tail -3; ls {OUT}/raw; head -5 {OUT}/raw/probe_kern_sum*.csv"),
        ("export", f"nsys export --type sqlite --output {OUT}/raw/probe_nsys.sqlite {OUT}/raw/probe_nsys.nsys-rep 2>&1 | tail -2; python -c \"import sqlite3; c=sqlite3.connect('{OUT}/raw/probe_nsys.sqlite'); print([r[0] for r in c.execute('select name from sqlite_master where type=\\\"table\\\" and name like \\\"CUPTI%\\\"')]); print(list(c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL limit 1'))); print([d[0] for d in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL limit 1').description]); print([d[0] for d in c.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY limit 1').description])\""),
    ]:
        rc, out, err = sh(cmd, timeout=900)
        print(f"$ [{name}] {cmd[:120]}\n{out}\n{err}\n", flush=True)
        report[name] = {"rc": rc, "out": out[-3000:], "err": err[-800:]}
else:
    log("nsys could not be installed")

json.dump(report, open(f"{OUT}/nsys_probe.json", "w"), indent=2)
finish("complete", {"nsys_path": nsys_path})
