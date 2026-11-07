import { useEffect, useState } from "react";
import { loadCustomers } from "./demoData";

// Loads the demo customers (features plus predicted effects) once.
export function useDemoCustomers() {
  const [state, setState] = useState({ customers: null, error: "" });

  useEffect(() => {
    let cancelled = false;
    loadCustomers()
      .then((customers) => {
        if (!cancelled) setState({ customers, error: "" });
      })
      .catch((e) => {
        if (!cancelled) setState({ customers: null, error: e.message });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}