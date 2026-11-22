// Decompile the functions named by a target list, for the npuc.h reconstruction (PORT-32 N3a step 4).
// Our own script. Its output is decompiled vendor code: it goes under output/ (git-ignored) and is never published.
//   analyzeHeadless <proj> <name> -process <binary> -noanalysis -scriptPath <dir> \
//     -postScript DecompileTargets.java <targets.txt> <out-dir>
// targets.txt, one per line (# comments):
//   str:<substring>   every function that references a defined string containing it
//   sym:<regex>       every function whose (mangled or demangled) name matches
//   addr:<hex>        the function containing that address
//   callers:<regex>   every function calling a function whose name matches
// Writes <out-dir>/<binary>/<entry>_<name>.c and an index.txt (entry, size, name, why).
//@category PORT-32

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.data.StringDataInstance;
import ghidra.program.model.listing.Data;
import ghidra.program.model.listing.DataIterator;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.symbol.Reference;
import ghidra.program.model.symbol.ReferenceIterator;
import ghidra.program.model.symbol.Symbol;

import java.io.File;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

public class DecompileTargets extends GhidraScript {
	private final Map<Function, String> picked = new LinkedHashMap<>();

	private void pick(Function f, String why) {
		if (f == null)
			return;
		picked.merge(f, why, (a, b) -> a.contains(b) ? a : a + "; " + b);
	}

	private void byString(String sub) throws Exception {
		DataIterator it = currentProgram.getListing().getDefinedData(true);
		while (it.hasNext() && !monitor.isCancelled()) {
			Data d = it.next();
			if (!d.hasStringValue())
				continue;
			String s = StringDataInstance.getStringDataInstance(d).getStringValue();
			if (s == null || !s.contains(sub))
				continue;
			ReferenceIterator refs = currentProgram.getReferenceManager().getReferencesTo(d.getAddress());
			while (refs.hasNext()) {
				Reference r = refs.next();
				pick(getFunctionContaining(r.getFromAddress()), "str \"" + sub + "\"");
			}
		}
	}

	private void bySymbol(String re, boolean callers) {
		Pattern p = Pattern.compile(re);
		FunctionIterator it = currentProgram.getFunctionManager().getFunctions(true);
		while (it.hasNext() && !monitor.isCancelled()) {
			Function f = it.next();
			String n = f.getName(true);
			boolean hit = p.matcher(n).find();
			// the mangled names stay as secondary labels at the entry
			for (Symbol s : currentProgram.getSymbolTable().getSymbols(f.getEntryPoint()))
				hit |= p.matcher(s.getName()).find();
			if (!hit)
				continue;
			if (!callers) {
				pick(f, "sym " + re);
				continue;
			}
			for (Function c : f.getCallingFunctions(monitor))
				pick(c, "calls " + n);
		}
	}

	@Override
	public void run() throws Exception {
		String[] a = getScriptArgs();
		List<String> lines = Files.readAllLines(new File(a[0]).toPath());
		File out = new File(a[1], currentProgram.getName());
		out.mkdirs();
		for (String l : lines) {
			l = l.strip();
			if (l.isEmpty() || l.startsWith("#"))
				continue;
			String k = l.substring(0, l.indexOf(':')), v = l.substring(l.indexOf(':') + 1);
			switch (k) {
			case "str": byString(v); break;
			case "sym": bySymbol(v, false); break;
			case "callers": bySymbol(v, true); break;
			case "addr": {
				Address ad = currentProgram.getAddressFactory().getDefaultAddressSpace().getAddress(Long.parseUnsignedLong(v.replace("0x", ""), 16));
				pick(getFunctionContaining(ad), "addr " + v);
				break;
			}
			default: printerr("unknown target " + l);
			}
		}
		DecompInterface di = new DecompInterface();
		DecompileOptions o = new DecompileOptions();
		di.setOptions(o);
		di.toggleCCode(true);
		di.toggleSyntaxTree(false);
		di.openProgram(currentProgram);
		try (PrintWriter idx = new PrintWriter(new File(out, "index.txt"))) {
			for (Map.Entry<Function, String> e : picked.entrySet()) {
				Function f = e.getKey();
				DecompileResults r = di.decompileFunction(f, 180, monitor);
				String name = f.getName(true).replaceAll("[^A-Za-z0-9_]+", "_");
				if (name.length() > 120)
					name = name.substring(0, 120);
				String fn = String.format("%s_%s.c", f.getEntryPoint().toString(), name);
				try (PrintWriter w = new PrintWriter(new File(out, fn))) {
					w.println("// " + f.getEntryPoint() + " " + f.getName(true) + " (" + f.getBody().getNumAddresses() + " bytes) <- " + e.getValue());
					for (Symbol s : currentProgram.getSymbolTable().getSymbols(f.getEntryPoint()))
						w.println("// label: " + s.getName());
					w.println(r.decompileCompleted() ? r.getDecompiledFunction().getC() : "// decompile failed: " + r.getErrorMessage());
				}
				idx.printf("%s\t%d\t%s\t%s%n", f.getEntryPoint(), f.getBody().getNumAddresses(), f.getName(true), e.getValue());
			}
		}
		println("DecompileTargets: " + picked.size() + " functions -> " + out);
	}
}
