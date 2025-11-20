export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="dashboard-layout">
      {/* Sidebar, Navbar, etc. */}
      {children}
    </div>
  );
}
