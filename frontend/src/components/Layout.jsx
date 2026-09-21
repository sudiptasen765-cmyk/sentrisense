import { NavLink, Outlet } from "react-router-dom";
import clsx from "clsx";

const NAV_ITEMS = [
  { to: "/", label: "Predict", end: true },
  { to: "/compare", label: "Compare" },
  { to: "/deployment", label: "Deployment" },
];

export default function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-ink-700">
        <div className="max-w-5xl mx-auto px-6 py-5 flex items-center justify-between">
          <div className="flex items-baseline gap-2">
            <span className="font-display text-xl tracking-tight">SentriSense</span>
            <span className="hidden sm:inline text-parchment-500 text-sm font-mono">
              sentiment, measured
            </span>
          </div>
          <nav className="flex gap-1">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  clsx(
                    "px-3 py-1.5 text-sm rounded-panel transition-colors",
                    isActive
                      ? "text-gold-500 bg-ink-900"
                      : "text-parchment-300 hover:text-parchment-100 hover:bg-ink-900"
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="flex-1">
        <div className="max-w-5xl mx-auto px-6 py-12 w-full">
          <Outlet />
        </div>
      </main>

      <footer className="border-t border-ink-700 mt-auto">
        <div className="max-w-5xl mx-auto px-6 py-6 text-xs text-parchment-500 font-mono flex justify-between">
          <span>SentriSense — IMDb sentiment analysis, serverless on AWS</span>
          <span>ap-south-1</span>
        </div>
      </footer>
    </div>
  );
}
