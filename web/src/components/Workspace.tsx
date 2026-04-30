import React from 'react';
import { Icons } from '../icons';
import { useAppStore } from '../store';
import s from './Workspace.module.css';

// Tab components
import { ZoneEditor }    from '../tabs/ZoneEditor';
import { POPSTab }       from '../tabs/POPSTab';
import { EventsTab }     from '../tabs/EventsTab';
import { AnalyticsTab }  from '../tabs/AnalyticsTab';
import { ThreeDViewTab } from '../tabs/ThreeDViewTab';
import { BirdsEyeTwoDTab } from '../tabs/BirdsEyeTwoDTab';
import { CaseReportTab } from '../tabs/CaseReportTab';
import { VideoInfoTab }  from '../tabs/VideoInfoTab';
import { ConfigTab }     from '../tabs/ConfigTab';
import { LegendTab }     from '../tabs/LegendTab';

type Group = 'tools' | 'reports' | 'reference';

interface TabDef {
  id: string;
  label: string;
  Icon: React.FC<{ size?: number }>;
  group: Group;
}

const TABS: readonly TabDef[] = [
  // Tools — interactive editors
  { id: 'zone',  label: 'Zone Editor',  Icon: Icons.zone,     group: 'tools' },

  // Reports — primary insights / data views
  { id: 'pops',  label: 'POPS',         Icon: Icons.user,     group: 'reports' },
  { id: 'event', label: 'Events',       Icon: Icons.pulse,    group: 'reports' },
  { id: 'anal',  label: 'Analytics',    Icon: Icons.chart,    group: 'reports' },
  { id: 'rep',   label: 'Case Report',  Icon: Icons.doc,      group: 'reports' },
  { id: '3d',    label: '3D View',      Icon: Icons.cube,     group: 'reports' },
  { id: '2d',    label: "Bird's-Eye 2D", Icon: Icons.eye,      group: 'reports' },

  // Reference — supporting data
  { id: 'info',  label: 'Video info',   Icon: Icons.cam,      group: 'reference' },
  { id: 'cfg',   label: 'Config',       Icon: Icons.settings, group: 'reference' },
  { id: 'leg',   label: 'Legend',       Icon: Icons.layers,   group: 'reference' },
] as const;

const GROUP_LABEL: Record<Group, string> = {
  tools:     'Tools',
  reports:   'Reports',
  reference: 'Reference',
};

export const Workspace: React.FC = () => {
  const { activeTab, setActiveTab, runResult, workspaceMode, setWorkspaceMode } = useAppStore();

  const highCount = runResult?.tracking_json?.summary?.high_priority ?? 0;
  const isFocus = workspaceMode === 'focus';

  // Build tab buttons grouped, with separators between groups
  const groups: Group[] = ['tools', 'reports', 'reference'];
  const tabRows = groups.map((g) => ({
    group: g,
    tabs: TABS.filter((t) => t.group === g),
  }));

  return (
    <div className={s.workspace}>
      {/* Tab bar */}
      <div className={s.tabs}>
        {tabRows.map(({ group, tabs }, gIdx) => (
          <React.Fragment key={group}>
            {gIdx > 0 && <div className={s.groupSep} aria-hidden="true" />}
            <span className={s.groupLabel} title={GROUP_LABEL[group]}>{GROUP_LABEL[group]}</span>
            {tabs.map(({ id, label, Icon }) => {
              const hasDot = id === 'event' && highCount > 0;
              const badge =
                id === 'pops' && runResult ? runResult.pops_carts.length :
                id === 'rep'  && runResult?.case_report_html ? 'AI' :
                undefined;
              return (
                <button
                  key={id}
                  className={`${s.tab} ${activeTab === id ? s.active : ''} ${s[`grp_${group}`] ?? ''}`}
                  onClick={() => setActiveTab(id)}
                >
                  <Icon size={12} />
                  {label}
                  {badge !== undefined && (
                    <span className={`${s.badge} ${activeTab === id ? s.badgeActive : ''}`}>{badge}</span>
                  )}
                  {hasDot && <span className={s.dot} />}
                </button>
              );
            })}
          </React.Fragment>
        ))}

        <div className={s.tabsTools}>
          <button
            className={s.iconBtn}
            onClick={() => setWorkspaceMode(isFocus ? 'split' : 'focus')}
            title={isFocus ? 'Show video output (split view)' : 'Maximize panel (hide video)'}
          >
            <Icons.expand size={12} />
          </button>
          <button className={s.iconBtn} title="More"><Icons.more size={12} /></button>
        </div>
      </div>

      {/* Tab panels */}
      <div className={s.panelWrap}>
        {activeTab === 'zone'  && <ZoneEditor />}
        {activeTab === 'pops'  && <POPSTab />}
        {activeTab === 'event' && <EventsTab />}
        {activeTab === 'anal'  && <AnalyticsTab />}
        {activeTab === '3d'    && <ThreeDViewTab />}
        {activeTab === '2d'    && <BirdsEyeTwoDTab />}
        {activeTab === 'rep'   && <CaseReportTab />}
        {activeTab === 'info'  && <VideoInfoTab />}
        {activeTab === 'cfg'   && <ConfigTab />}
        {activeTab === 'leg'   && <LegendTab />}
      </div>
    </div>
  );
};
