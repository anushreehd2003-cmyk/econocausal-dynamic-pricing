// Observed purchase rate per discount level from the uploaded campaign data.
export function purchaseRateByDiscount(rows) {
  return [0, 10, 20].map((d) => {
    const group = rows.filter((r) => r.discount === d);
    const buyers = group.filter((r) => r.purchased === 1).length;
    return {
      level: d,
      n: group.length,
      rate: group.length ? buyers / group.length : 0,
    };
  });
}