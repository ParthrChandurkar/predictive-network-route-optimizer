import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://predictive-network-route-optimizer.vercel.app"),
  title: "NetRoute Intelligence | Predictive Network Degradation",
  description:
    "An evidence-led network experimentation platform for LTE degradation detection and risk-aware routing.",
  openGraph: {
    title: "NetRoute Intelligence",
    description:
      "Real LTE telemetry, degradation-aware routing, controlled Mininet experiments, and observable decisions.",
    type: "website",
  },
  alternates: {
    canonical: "/",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
