import React, { useState } from 'react';
import PatientRiskPredictor from './pages/PatientRiskPredictor';

export const CHART_STYLES = {
  fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  fontSize: 12,
};

const Icons = {
  BarChart2: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>,
  Users: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>,
  Activity: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>,
  AlertTriangle: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>,
  Zap: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>,
  Radio: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="2"></circle><path d="M16.24 7.76a6 6 0 0 1 0 8.49m-8.48-.01a6 6 0 0 1 0-8.49m11.31-2.82a10 10 0 0 1 0 14.14m-14.14 0a10 10 0 0 1 0-14.14"></path></svg>,
  Target: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>
};

const NAV = [
  { id: 6, label: 'Patient Risk Predictor', icon: Icons.Radio },
];

export default function App() {
  const [page, setPage] = useState(6);
  const [isCollapsed, setIsCollapsed] = useState(false);

  const pages = {
    6: <PatientRiskPredictor />,
    7: <PatientRiskPredictor /> // Fallback placeholder
  };

  return (
    <div style={S.root}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: ${CHART_STYLES.fontFamily}; background: #eef2f2; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #e0eaea; }
        ::-webkit-scrollbar-thumb { background: #a8c8c4; border-radius: 4px; }
        @keyframes fadeIn { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
        .page-enter { animation: fadeIn 0.35s ease forwards; }
        select, select option { background:#fff; color:#1a3a3a; }
        input[type=range] { accent-color: #0d7a6b; cursor: pointer; }
      `}</style>

      <aside style={{ ...S.sidebar, width: isCollapsed ? 80 : 260 }}>
        <div style={{ ...S.brand, padding: isCollapsed ? '24px 10px' : '24px 16px', flexDirection: isCollapsed ? 'column' : 'row', gap: isCollapsed ? 20 : 12 }}>
          <div style={S.brandIcon}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
          </div>
          {!isCollapsed && (
            <div style={{ flex: 1, minWidth: 0, overflow: 'hidden' }}>
              <div style={S.brandTitle}>Stroke Analytics</div>
              <div style={S.brandSub}>Awareness Dashboard</div>
            </div>
          )}
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            style={{ ...S.collapseBtn, alignSelf: isCollapsed ? 'center' : 'auto' }}
            title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {isCollapsed ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="13 17 18 12 13 7"></polyline><polyline points="6 17 11 12 6 7"></polyline></svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="11 17 6 12 11 7"></polyline><polyline points="18 17 13 12 18 7"></polyline></svg>
            )}
          </button>
        </div>

        <div style={{ width: '100%', height: 1, background: 'rgba(255,255,255,0.05)' }} />



        <nav style={{ flex: 1, marginTop: isCollapsed ? 20 : 0 }}>
          {NAV.map(n => {
            const active = page === n.id;
            return (
              <button key={n.id} onClick={() => setPage(n.id)} style={{
                ...S.navBtn,
                background: active ? '#11615a' : 'transparent',
                borderRadius: '0 8px 8px 0',
                marginRight: isCollapsed ? 0 : '24px',
                marginBottom: '8px',
                padding: isCollapsed ? '16px 0' : '14px 20px 14px 24px',
                justifyContent: isCollapsed ? 'center' : 'flex-start',
                position: 'relative'
              }} title={isCollapsed ? n.label : undefined}>

                {active && !isCollapsed && (
                  <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 4, background: '#14b8a6', borderRadius: '0 4px 4px 0' }} />
                )}
                {active && isCollapsed && (
                  <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 4, background: '#14b8a6', borderRadius: '0 4px 4px 0' }} />
                )}

                <span style={{
                  display: 'flex',
                  alignItems: 'center',
                  fontSize: 20,
                  color: active ? '#ffffff' : '#b2d1cd',
                  transition: 'color 0.2s'
                }}>
                  {n.icon}
                </span>

                {!isCollapsed && (
                  <span style={{
                    fontSize: 14,
                    marginLeft: 16,
                    fontWeight: active ? 600 : 500,
                    color: active ? '#ffffff' : '#b2d1cd',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    transition: 'color 0.2s',
                    flex: 1
                  }}>
                    {n.label}
                  </span>
                )}
              </button>
            );
          })}
        </nav>

        <div style={{ ...S.sidebarFooter, padding: isCollapsed ? '16px 5px' : '16px 24px', textAlign: isCollapsed ? 'center' : 'left' }}>
          {!isCollapsed ? (
            <div style={S.modelChip}>
              <span style={{ color: '#14b8a6', fontSize: 8, marginRight: 6 }}>●</span>Random Forest · AUC 0.993
            </div>
          ) : (
            <div style={{ ...S.modelChip, padding: '6px' }}>
              <span style={{ color: '#14b8a6', fontSize: 8 }}>●</span>
            </div>
          )}
        </div>
      </aside>

      <main style={S.main}>
        <div key={page} className="page-enter">{pages[page]}</div>
      </main>
    </div>
  );
}

const S = {
  root: { display: 'flex', height: '100vh', background: '#eef2f2', color: '#1a3a3a', overflow: 'hidden' },
  sidebar: { background: '#0d322d', borderRight: 'none', display: 'flex', flexDirection: 'column', flexShrink: 0, padding: '0 0 24px 0', boxShadow: '2px 0 10px rgba(0,0,0,0.1)', transition: 'width 0.3s cubic-bezier(0.4, 0, 0.2, 1)', zIndex: 10 },
  brand: { display: 'flex', alignItems: 'center', boxSizing: 'border-box', width: '100%' },
  brandIcon: { width: 40, height: 40, background: '#14b8a6', borderRadius: 11, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#ffffff', flexShrink: 0, boxShadow: '0 0 16px rgba(20, 184, 166, 0.4)' },
  brandTitle: { fontSize: 16, fontWeight: 700, color: '#ffffff', letterSpacing: 0.2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' },
  brandSub: { fontSize: 12, color: '#719d96', letterSpacing: 0.2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', marginTop: 2 },
  collapseBtn: { background: 'transparent', border: '1px solid rgba(255,255,255,0.06)', outline: 'none', color: '#b2d1cd', cursor: 'pointer', width: 32, height: 32, display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: '8px', flexShrink: 0, transition: 'background 0.2s, border 0.2s' },
  navLabel: { fontSize: 12, fontWeight: 700, color: '#5e8782', letterSpacing: 1.2, padding: '32px 24px 12px', textTransform: 'uppercase' },
  navBtn: { width: '100%', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', transition: 'all 0.2s', textAlign: 'left', outline: 'none' },
  sidebarFooter: { marginTop: 'auto', borderTop: '1px solid rgba(255,255,255,0.05)' },
  modelChip: { background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: 6, padding: '8px 12px', fontSize: 11, color: '#c0d6d3', whiteSpace: 'nowrap', overflow: 'hidden' },
  main: { flex: 1, overflow: 'auto', background: '#eef2f2' },
};
