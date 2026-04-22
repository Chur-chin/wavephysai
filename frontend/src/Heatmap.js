import React from "react";

export default function Heatmap({ data }) {
  if (!data) return null;

  return (
    <div style={{ display: "grid", gridTemplateColumns: `repeat(${data.length}, 4px)` }}>
      {data.flat().map((v, i) => {
        const color = `rgb(${Math.min(255, Math.abs(v)*500)},0,0)`
        return <div key={i} style={{ width: 4, height: 4, background: color }} />
      })}
    </div>
  );
}
