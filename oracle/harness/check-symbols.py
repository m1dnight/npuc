#!/usr/bin/env python3
"""Static stand-in for Bionic's RTLD_NOW binding, on the host (no emulator), with Bionic's lookup scopes.

  check-symbols.py <llvm-readelf> <run-dir> <process>...
  <process> = <executable>[:<dlopen root>[,<dlopen root>...]]   e.g.  npuc-probe:libnpuc_api.so  npuc-conv

For each process: the executable and its DT_NEEDED tree load first, then each dlopen root in order. Every library
is relocated once, when it is first loaded, and each of its undefined dynamic symbols (with its version, when it
has one) must be defined in its lookup scope, as Bionic's linker (API >= 23) searches it:
  global group  the executable only (LD_PRELOAD and RTLD_GLOBAL additions are ignored: the phone showed that a
                library dlopen()ed RTLD_GLOBAL earlier did not satisfy a later dlopen, 2026-11-21 run 104734Z)
  local group   breadth-first DT_NEEDED closure of the root that loaded it (the executable, or the dlopen root)
So an underlinked library (one that uses a symbol of a library it does not list in DT_NEEDED) passes only when it
is loaded under a root whose closure contains the provider: the case of libnpuc_graph (Logging::NamedPoint, from
libnpuc_common) and libnpuc_backend (NPU typeinfo, from libnpuc_cmdq), which work under libnpuc_api's closure and
fail when dlopen()ed one by one. Weak undefined symbols may stay unbound. linker64 provides ld-android.so's symbols.
Exit 0 when every process binds completely; prints the missing symbols per library and process otherwise."""
import os, re, subprocess, sys

ROW = re.compile(r"^\s*\d+:\s+([0-9a-f]+)\s+\d+\s+(\w+)\s+(\w+)\s+(\w+)\s+(\S+)\s+(\S+)$")


def dynsyms(readelf, path):
    out = subprocess.run([readelf, "-W", "--dyn-syms", path], capture_output=True, text=True, check=True).stdout
    defs, undefs = set(), []
    for line in out.splitlines():
        m = ROW.match(line)
        if not m:
            continue
        _val, typ, bind, vis, ndx, name = m.groups()
        if not name or typ == "SECTION" or name == "UND":
            continue
        base, _, ver = name.partition("@")
        ver = ver.lstrip("@")
        if ndx == "UND":
            undefs.append((base, ver, bind))
        elif bind in ("GLOBAL", "WEAK") and vis != "HIDDEN":
            defs.add((base, ver))
            defs.add((base, ""))
    return defs, undefs


def needed(readelf, path):
    out = subprocess.run([readelf, "-d", path], capture_output=True, text=True, check=True).stdout
    return re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", out)


class Dir:
    def __init__(self, readelf, root):
        self.readelf, self.root, self.cache = readelf, root, {}

    def elf(self, name):
        if name not in self.cache:
            path = os.path.join(self.root, name)
            if not os.path.isfile(path):
                self.cache[name] = None
            else:
                d, u = dynsyms(self.readelf, path)
                if name == "ld-android.so":  # the stub only declares them; linker64 defines them
                    d |= dynsyms(self.readelf, os.path.join(self.root, "linker64"))[0]
                self.cache[name] = (d, u, needed(self.readelf, path))
        return self.cache[name]

    def closure(self, root):
        """Breadth-first DT_NEEDED closure, root first (Bionic's local group order); missing names kept."""
        order, seen, queue = [], {root}, [root]
        while queue:
            n = queue.pop(0)
            order.append(n)
            e = self.elf(n)
            for c in (e[2] if e else []):
                if c not in seen:
                    seen.add(c)
                    queue.append(c)
        return order


def check_process(d, exe, roots):
    bad, loaded, lines = 0, set(), []
    for root in [exe] + roots:
        group = d.closure(root)
        scope = set()
        for n in ([exe] if root != exe else []) + group:
            e = d.elf(n)
            if e:
                scope |= e[0]
        for n in group:
            if n in loaded:
                continue
            loaded.add(n)
            e = d.elf(n)
            if e is None:
                lines.append(f"  {n}: NOT IN THE DIRECTORY (needed under {root})")
                bad += 1
                continue
            _defs, undefs, _need = e
            miss = sorted({f"{b}@{v}" if v else b for b, v, bind in undefs if bind != "WEAK" and (b, v) not in scope})
            weak = sum(1 for b, v, bind in undefs if bind == "WEAK" and (b, v) not in scope)
            how = "loaded with the executable" if root == exe else f"loaded under dlopen({root})"
            lines.append(f"  {n}: {len(undefs)} imports, {'OK' if not miss else 'MISSING'}, {how}"
                         + (f", {weak} weak unbound" if weak else ""))
            lines += [f"    {m}" for m in miss]
            bad += len(miss)
    return bad, lines


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    d = Dir(sys.argv[1], sys.argv[2])
    total = 0
    for spec in sys.argv[3:]:
        exe, _, rest = spec.partition(":")
        roots = [r for r in rest.split(",") if r]
        bad, lines = check_process(d, exe, roots)
        print(f"process {exe}" + (f", dlopen {', '.join(roots)}" if roots else "") + (": OK" if not bad else f": {bad} missing"))
        print("\n".join(lines))
        total += bad
    print("check-symbols: " + ("PASS" if not total else f"FAIL ({total} missing)"))
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
