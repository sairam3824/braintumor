import { useState, useEffect } from "react";
import HistoryTable from "../components/HistoryTable.jsx";

export default function DashboardPage() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const stored = JSON.parse(localStorage.getItem("prediction_history") || "[]");
    setHistory(stored);
  }, []);

  const clearHistory = () => {
    localStorage.removeItem("prediction_history");
    setHistory([]);
  };

  // Compute stats
  const totalScans = history.length;
  const tumorsFound = history.filter((h) => h.is_tumor).length;
  const clearScans = totalScans - tumorsFound;
  const avgConfidence =
    totalScans > 0
      ? (history.reduce((sum, h) => sum + h.confidence, 0) / totalScans).toFixed(1)
      : "—";

  const stats = [
    { value: totalScans, label: "Total Scans" },
    { value: tumorsFound, label: "Tumors Found" },
    { value: clearScans, label: "Clear Scans" },
    { value: `${avgConfidence}%`, label: "Avg Confidence" },
  ];

  return (
    <div className="page-container" id="dashboard-page">
      <header className="page-header">
        <h1 className="page-title">Dashboard</h1>
        <p className="page-subtitle">
          Track your scan history and prediction statistics at a glance.
        </p>
      </header>

      {/* Stats Cards */}
      <div className="stats-grid">
        {stats.map((s) => (
          <div key={s.label} className="stat-card glass-card">
            <div className="stat-value">{s.value}</div>
            <div className="stat-label">{s.label}</div>
          </div>
        ))}
      </div>

      {/* History Table */}
      <HistoryTable history={history} onClear={clearHistory} />
    </div>
  );
}
