import os
import sys

base_source_dirs = {
    "mathias-thinkpad": "/home/mathias/dev",
    "mathias-desktop": "/home/mathias/dev/ist",
}
base_move_dirs = {
    "mathias-thinkpad": "/home/mathias/dev/azure",
    "deephawk": "/home/mathias/",
    "mathias-desktop": "/home/mathias/dev/azure/",
}

base_target_dirs = {
    "deephawk": "deephawk/dev/",
    "pixie": "deephawk/dev/azure/pixie",
    "hpc": "hpc",
    "z1": "deephawk/dev/azure/z1",
    "z2": "deephawk/dev/azure/z2",
    "rpc": "rpc",
    "1080": "deephawk/dev/azure/1080",
    "deepart": "deephawk/dev/azure/deepart",
}

sync_dirs = [
    "s4",
    "s4/configs/experiment",
    "s4/configs/model/layer",
    "s4/src/models/hippo",
    "s4/src/models/sequence/ss",
]
sync_ext = [".py", ".sh", ".sl", ".yaml"]


host = os.uname()[1].lower()
if not host in base_move_dirs or not host in base_move_dirs:
    print("Unknown host: " + host)
    sys.exit(-1)

target = "deephawk"
if len(sys.argv) > 1:
    target = sys.argv[1].lower()
if not target in base_target_dirs:
    print("Unknown target: " + target)
    sys.exit(-1)

flags = "ruv"
if len(sys.argv) > 2:
    if sys.argv[2].lower() == "--force":
        flags = "rv"
        print("Force rewrite")

print('Syncing from "' + host + '" to "' + target + '"')
for sync_dir in sync_dirs:
    print('   moving dir "' + sync_dir + '"')
    source_dir = os.path.join(base_source_dirs[host], sync_dir)

    target_dir = os.path.join(base_move_dirs[host], base_target_dirs[target], sync_dir)

    print("Source dir: " + str(source_dir))
    print("Target dir: " + str(target_dir))
    cmd_str = (
        "rsync -"
        + flags
        + " '"
        + source_dir
        + "/' --exclude '.git' --include '*.py' --include '*.yaml' --include '*.sl' --include '*.sh' --exclude '*'  '"
        + target_dir
        + "/'"
    )

    os.system(cmd_str)
    # print(cmd_str)