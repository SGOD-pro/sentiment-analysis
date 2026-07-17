import * as React from "react"
import {
  Sidebar,
  SidebarContent,
} from "@/components/ui/sidebar"

export function AppSidebar({
  children,
  ...props
}: React.ComponentProps<typeof Sidebar>) {
  return (
    <Sidebar variant="inset" className="pt-14" {...props}>
      <SidebarContent>
        {children}
      </SidebarContent>
    </Sidebar>
  )
}
