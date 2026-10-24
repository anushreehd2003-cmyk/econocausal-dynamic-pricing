import Papa from "papaparse";

export async function loadQini() {
  const res = await fetch("/demo/qini_curves.json");
  try {
    if (!res.ok) throw new Error();
    return await res.json();
  } catch {
    throw new Error("Could not load /demo/qini_curves.json");
  }
}

export async function loadCustomers() {
  const res = await fetch("/demo/customers_ite.csv");
  const text = res.ok ? await res.text() : "";
  // A missing file in the Vite dev server returns the HTML page, not an error
  if (!text || text.trimStart().startsWith("<")) {
    throw new Error("Could not load /demo/customers_ite.csv");
  }
  const parsed = Papa.parse(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });
  return parsed.data;
}