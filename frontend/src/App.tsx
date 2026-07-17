import { useCallback, useState } from "react";
import { Link, Outlet, Route, Routes, useLocation } from "react-router-dom";
import { Toaster } from "@/components/ui/sonner";
import { Button } from "@/components/ui/button";
import { Moon, Sun, Upload, BarChart3, FileText, Bell, Settings2 } from "lucide-react";
import Dashboard from "./pages/Dashboard";
import Reviews from "./pages/Reviews";
import Reports from "./pages/Reports";
import Settings from "./pages/Settings";
import UploadPage from "./pages/Upload";
import { SidebarProvider } from "@/components/ui/sidebar";

function useTheme() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  const toggle = useCallback(() => {
    const next = !dark;
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("theme", next ? "dark" : "light");
    setDark(next);
  }, [dark]);
  return { dark, toggle };
}

const NAV = [
  { to: "/", label: "Dashboard", icon: BarChart3 },
  { to: "/reviews", label: "Reviews", icon: FileText },
  { to: "/reports", label: "Reports", icon: BarChart3 },
];

function TopBar() {
  const { dark, toggle } = useTheme();
  const { pathname } = useLocation();
  return (
    <header className="sticky top-0 inset-x-0 h-14 z-50 flex items-center justify-between px-6 bg-sidebar">
      {/* Logo + Nav */}
      <div className="flex items-center gap-8">
        <span className="text-base font-bold tracking-tight">SentiMetric</span>
        <nav className="hidden md:flex items-center gap-1">
          {NAV.map(({ to, label }) => {
            const active = pathname === to;
            return (
              <Link key={to} to={to}
                className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${active
                  ? "text-primary border-b-2 border-primary rounded-none pb-3.5"
                  : "text-muted-foreground hover:text-foreground hover:bg-accent"}`}>
                {label}
              </Link>
            );
          })}
        </nav>
      </div>
      {/* Right controls */}
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="sm" asChild>
          <Link to="/settings"><Settings2 className="w-3.5 h-3.5" /></Link>
        </Button>
        <Button variant="outline" size="sm" asChild>
          <Link to="/upload"><Upload className="w-3.5 h-3.5 mr-1.5" />Import Data</Link>
        </Button>
        <Button variant="ghost" size="sm" className="text-muted-foreground"><Bell className="w-4 h-4" /></Button>
        <Button variant="ghost" size="icon" onClick={toggle} className="text-muted-foreground">
          {dark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </Button>
      </div>
    </header>
  );
}


function DashboardLayout() {
  return (
    <>
      <TopBar />
      <SidebarProvider className="min-h-0 h-[calc(100svh-3.5rem)]">
        <Outlet />
      </SidebarProvider>
    </>
  );
}

export default function App() {
  return (
    <>
      <Routes>
        {/* Dashboard pages – shared top nav */}
        <Route element={<DashboardLayout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/reviews" element={<Reviews />} />
          <Route path="/reports" element={<Reports />} />
        </Route>
        {/* Standalone pages – no top nav */}
        <Route path="/settings" element={<Settings />} />
        <Route path="/upload" element={<UploadPage />} />
      </Routes>
      <Toaster richColors position="top-right" />
    </>
  );
}
