import React, { useEffect, useState } from "react";
import Heatmap from "./Heatmap";

function App() {
  const [frames, setFrames] = useState([]);
  const [signal, setSignal] = useState([]);
  const [i, setI] = useState(0);

  useEffect(() => {
    fetch("http://localhost:8000/data")
      .then(res => res.json())
      .then(data => {
        setFrames(data.frames);
        setSignal(data.signal);
      });
  }, []);

  useEffect(() => {
    const id = setInterval(() => {
      setI(prev => (frames.length ? (prev + 1) % frames.length : 0));
    }, 50);

    return () => clearInterval(id);
  }, [frames]);

  return (
    <div>
      <h2>Wave Field</h2>
      <Heatmap data={frames[i]} />

      <h2>Signal</h2>
      <div>{signal[i]}</div>
    </div>
  );
}

export default App;
