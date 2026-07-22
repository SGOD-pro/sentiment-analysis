import { AppSidebar } from "@/components/app-sidebar"
import { Separator } from "@/components/ui/separator"
import {
  SidebarInset,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import type React from "react"

interface DashboardPageProps {
  sidebar: React.ReactNode
  children: React.ReactNode
}

export function DashboardPage({ sidebar, children }: DashboardPageProps) {
  return (
    <>
      <AppSidebar>{sidebar}</AppSidebar>
      <SidebarInset className="h-[calc(100svh-4.5rem)] overflow-hidden">
        <header className="flex h-12 shrink-0 items-center gap-2 px-4 bg-background/40 backdrop-blur-3xl">
          <SidebarTrigger className="-ml-1" />
          <Separator orientation="vertical" className="mr-2 data-[orientation=vertical]:h-4" />
        </header>
        <div className="flex flex-1 flex-col gap-4 p-4 pt-0 overflow-y-auto min-h-0">
          {children}
        </div>
      </SidebarInset>
    </>
  )
}
