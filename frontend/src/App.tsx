import { useCallback, useState } from "react";
import { Link, Outlet, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { Toaster } from "@/components/ui/sonner";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Moon, Sun, Upload, BarChart3, FileText, Bell, Cog, RotateCcw } from "lucide-react";
import Dashboard from "./pages/Dashboard";
import Reviews from "./pages/Reviews";
import Reports from "./pages/Reports";
import Settings from "./pages/Settings";
import UploadPage from "./pages/Upload";
import { SidebarProvider } from "@/components/ui/sidebar";
import { useSessionStore } from "@/hooks/useSessionStore";
import GooeyNav from "./components/GooeyNav";

function useTheme() {
  const [dark, setDark] = useState(() => {
    const saved = localStorage.getItem("theme");
    if (saved) return saved === "dark";
    return document.documentElement.classList.contains("dark");
  });

  const toggle = useCallback(() => {
    const next = !dark;
    
    const switchTheme = () => {
      // 1. Synchronously update the DOM and localStorage
      document.documentElement.classList.toggle("dark", next);
      localStorage.setItem("theme", next ? "dark" : "light");
      // 2. Trigger React re-render
      setDark(next);
    };

    // 3. Use View Transitions API if available, otherwise fallback gracefully
    if (!document.startViewTransition) {
      switchTheme();
    } else {
      document.startViewTransition(switchTheme);
    }
  }, [dark]);

  return { dark, toggle };
}

interface GooeyNavItem {
  label: string;
  href: string;
}

const NAV: GooeyNavItem[] = [
  { href: "/", label: "Dashboard" },
  { href: "/reviews", label: "Reviews" },
  { href: "/reports", label: "Reports" },
];

function TopBar() {
  const { dark, toggle } = useTheme();
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const clearSession = useSessionStore((s) => s.clearSession);
  const [confirmOpen, setConfirmOpen] = useState(false);

  const doNewSession = () => {
    clearSession();
    setConfirmOpen(false);
    navigate("/upload");
  };

  const handleNewSession = () => {
    // If already on upload page, just do it. Otherwise confirm.
    if (pathname === "/upload") {
      doNewSession();
    } else {
      setConfirmOpen(true);
    }
  };

  return (
    <>
      <header className="sticky top-0 inset-x-0 h-14 z-50 flex items-center justify-between px-6 bg-sidebar">
        {/* Logo + Nav */}
        <div className="flex items-center gap-8">
          <span className="text-base font-bold tracking-tight">SentiX</span>
          <nav className="hidden md:flex items-center gap-1">
            <div className="h-full overflow-hidden">
              <GooeyNav
                items={NAV}
                particleCount={10}
                particleDistances={[90, 10]}
                particleR={100}
                initialActiveIndex={0}
                animationTime={400}
                timeVariance={200}
                colors={[1, 2, 3, 1, 2, 3, 1, 4]}
              />
            </div>
            {/* {NAV.map(({ href, label }) => {
              const active = pathname === to;
              return (
                <Link key={to} to={to}
                  className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${active
                    ? "text-primary border-b-2 border-primary rounded-none pb-3.5"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent"}`}>
                  {label}
                </Link>
              );
            })} */}
          </nav>
        </div>
        {/* Right controls */}
        <div className="flex items-center gap-2">

          <Button variant="ghost" size="sm" onClick={handleNewSession} className="text-muted-foreground gap-1.5">
            <RotateCcw className="w-3.5 h-3.5" />New Session
          </Button>
          <Button variant="outline" size="sm" asChild>
            <Link to="/upload"><Upload className="w-3.5 h-3.5 mr-1.5" />Import Data</Link>
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate("/settings")
            }>
            <span>
              <Cog className="w-3.5 h-3.5 text-muted-foreground" />
            </span>
          </Button>
          <Button variant="ghost" size="sm" className="text-muted-foreground">
            <Bell className="w-4 h-4" />
          </Button>
          <Button variant="ghost" size="icon" onClick={toggle} className="text-muted-foreground">
            {dark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </Button>
        </div>
      </header>

      {/* Confirm dialog for New Session when not on /upload */}
      <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <DialogContent className="min-w-72 max-w-2xl">
          <DialogHeader>
            <DialogTitle>Start New Session?</DialogTitle>
            <DialogDescription>
              This will leave your current dashboard view and reset the session. Your uploaded data stays safe in storage — you can re-upload the same file anytime.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2">
            <Button variant="outline" onClick={() => setConfirmOpen(false)}>Cancel</Button>
            <Button onClick={doNewSession}>Start New Session</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
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
