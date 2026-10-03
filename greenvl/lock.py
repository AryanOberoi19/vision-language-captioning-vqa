"""One process at a time per run directory: a second copy of the same run would corrupt its checkpoints."""
import atexit
import os
from pathlib import Path


class RunLock:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                pid = int(self.path.read_text().strip() or 0)
            except (OSError, ValueError):
                pid = 0
            if pid and _alive(pid):
                raise SystemExit(f"{self.path.parent} is already in use by process {pid}. Stop that process first "
                                 f"(or, if it no longer exists, delete {self.path}).")
            self.path.unlink(missing_ok=True)  # stale lock left by a process that has ended
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        atexit.register(self.release)

    def release(self):
        try:
            if self.path.exists() and self.path.read_text().strip() == str(os.getpid()):
                self.path.unlink()
        except OSError:
            pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
