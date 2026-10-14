import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: '50 Startups: Multiple Linear Regression & Feature Selection Explorer',
  description: 'AIoT HW2 - Interactive CRISP-DM Machine Learning Dashboard for Multiple Linear Regression, RFE, SFS, Statistical Inference, Live Profit Predictor, and Optuna Optimization',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-TW">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet" />
      </head>
      <body className="bg-slate-50 text-slate-800 antialiased min-h-screen">
        {children}
      </body>
    </html>
  );
}
