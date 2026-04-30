import React, { useEffect } from 'react';
import { TopBar }        from './components/TopBar';
import { Rail }          from './components/Rail';
import { TrackedOutput } from './components/TrackedOutput';
import { Workspace }     from './components/Workspace';
import { JsonDock }      from './components/JsonDock';
import { TweaksPanel }   from './components/TweaksPanel';
import { useTweaksStore, useAppStore } from './store';
import s from './App.module.css';

export const App: React.FC = () => {
  const { tweaks } = useTweaksStore();
  const { dockOpen, setDockOpen, workspaceMode } = useAppStore();

  const showDock = tweaks.showDock && dockOpen;
  const focusMode = workspaceMode === 'focus';

  useEffect(() => {
    document.documentElement.style.setProperty('--accent-h', String(tweaks.accentHue));
    document.documentElement.dataset.density = tweaks.density;
  }, [tweaks.accentHue, tweaks.density]);

  return (
    <div
      className={s.app}
      data-rail-side={tweaks.railSide}
      data-dock={showDock ? 'visible' : 'hidden'}
      data-focus={focusMode ? 'true' : 'false'}
    >
      <TopBar dockOpen={showDock} onDockToggle={() => setDockOpen(!dockOpen)} />
      <Rail />

      <main className={s.main}>
        {!focusMode && (
          <>
            <TrackedOutput tweaks={tweaks} />
            <div className={s.splitter} />
          </>
        )}
        <Workspace />
      </main>

      {showDock && <JsonDock />}
      <TweaksPanel />
    </div>
  );
};
