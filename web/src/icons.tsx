// Inline-SVG icon set — ported from design_handoff_gk_pops/icons.jsx
import React from 'react';

interface IcoProps {
  size?: number;
  fill?: string;
  stroke?: string;
  sw?: number;
  viewBox?: string;
  d?: string | React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}

const Ico: React.FC<IcoProps> = ({
  d, size = 14, fill, stroke = 'currentColor', sw = 1.5,
  viewBox = '0 0 24 24', className, style,
}) => (
  <svg
    width={size} height={size} viewBox={viewBox}
    fill={fill || 'none'}
    stroke={fill ? 'none' : stroke}
    strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round"
    className={className} style={style}
  >
    {typeof d === 'string' ? <path d={d} /> : d}
  </svg>
);

type IconFC = React.FC<{ size?: number; className?: string; style?: React.CSSProperties }>;

const mk = (d: string, extra?: Partial<IcoProps>): IconFC =>
  ({ size, className, style }) => <Ico d={d} size={size} className={className} style={style} {...extra} />;

export const Icons = {
  upload:    mk("M12 16V4m0 0l-4 4m4-4l4 4M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2"),
  play:      mk("M6 4l14 8-14 8V4z", { fill: 'currentColor' }),
  pause:     mk("M7 5h3v14H7zM14 5h3v14h-3z", { fill: 'currentColor' }),
  search:    mk("M11 19a8 8 0 100-16 8 8 0 000 16zm0 0l-3 3m12 0l-5-5"),
  bolt:      mk("M13 2L4 14h6l-1 8 9-12h-6l1-8z"),
  layers:    mk("M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5M3 18l9 5 9-5"),
  pin:       mk("M12 21V12m0 0a4 4 0 100-8 4 4 0 000 8z"),
  poly:      mk("M5 7l7-4 7 4-2 10-5 4-5-4L5 7z"),
  rect:      mk("M4 5h16v14H4z"),
  cursor:    mk("M5 3l7 17 2-7 7-2L5 3z"),
  hand:      mk("M9 11V5a2 2 0 014 0v6m0-3a2 2 0 014 0v5m0-3a2 2 0 014 0v6c0 4-3 7-7 7s-7-2-9-6l-3-6a2 2 0 013-2l3 4"),
  trash:     mk("M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"),
  undo:      mk("M9 14L4 9l5-5M4 9h11a5 5 0 010 10h-3"),
  check:     mk("M5 12l5 5L20 7"),
  x:         mk("M6 6l12 12M6 18L18 6"),
  download:  mk("M12 4v12m0 0l-4-4m4 4l4-4M4 20h16"),
  copy:      mk("M9 9h11v11H9zM4 4h11v3M4 4v11h3"),
  expand:    mk("M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"),
  alert:     mk("M12 9v4m0 4h.01M3.6 18.5L11 5.5a1.2 1.2 0 012 0l7.4 13a1.2 1.2 0 01-1 1.7H4.6a1.2 1.2 0 01-1-1.7z"),
  eye:       mk("M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12zm10 3a3 3 0 100-6 3 3 0 000 6z"),
  user:      mk("M12 12a4 4 0 100-8 4 4 0 000 8zM4 21a8 8 0 0116 0"),
  settings:  mk("M12 9a3 3 0 100 6 3 3 0 000-6zm9 3l-2-1.2-.4-1.5 1.1-2-1.4-1.4-2 1.1L15 6.4 13.8 4h-2L11 6.4l-1.5.4-2-1.1L6.1 7.1l1.1 2-.4 1.5L4.5 12l1.2 1.2.4 1.5-1.1 2 1.4 1.4 2-1.1L10 17.6 11 20h2l1-2.4 1.5-.4 2 1.1 1.4-1.4-1.1-2 .4-1.5L21 12z"),
  filter:    mk("M4 5h16l-6 8v6l-4-2v-4L4 5z"),
  more:      mk("M12 6h.01M12 12h.01M12 18h.01", { sw: 3 }),
  link:      mk("M10 14a5 5 0 007.5.5l3-3a5 5 0 00-7-7l-1 1m-2 8a5 5 0 00-7.5-.5l-3 3a5 5 0 007 7l1-1"),
  cam:       mk("M4 7h3l2-2h6l2 2h3v12H4V7zm8 9a4 4 0 100-8 4 4 0 000 8z"),
  doc:       mk("M8 3h8l4 4v14H4V3h4zm6 0v5h6M8 13h8M8 17h8M8 9h2"),
  chart:     mk("M4 20V10m6 10V4m6 16v-8m6 8V8"),
  cube:      mk("M12 3l9 5v8l-9 5-9-5V8l9-5zm0 0v18M3 8l9 5 9-5"),
  sparkles:  mk("M12 3v4m0 10v4m9-9h-4m-10 0H3m14.5-6.5l-3 3m-7 7l-3 3m13 0l-3-3m-7-7l-3-3"),
  refresh:   mk("M4 12a8 8 0 0114-5l2 2m0-5v5h-5M20 12a8 8 0 01-14 5l-2-2m0 5v-5h5"),
  history:   mk("M3 12a9 9 0 109-9 9 9 0 00-7 3.5M3 4v4h4M12 7v5l4 2"),
  pulse:     mk("M3 12h4l2-7 4 14 2-7h6"),
  zone:      mk("M3 8l5-5h8l5 5v8l-5 5H8l-5-5V8z"),
  cart:      mk("M3 4h2l2 12h12l2-8H6m1 14a1 1 0 100-2 1 1 0 000 2zm10 0a1 1 0 100-2 1 1 0 000 2z"),
  arrowRight:mk("M5 12h14m0 0l-6-6m6 6l-6 6"),
  flag:      mk("M5 21V4m0 0h12l-3 5 3 5H5"),
};
