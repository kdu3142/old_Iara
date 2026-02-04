import { useMemo, useRef, useState, useEffect } from "react";
import { createPortal } from "react-dom";
import { RTVIEvent, type PipecatMetricData } from "@pipecat-ai/client-js";
import { useRTVIClientEvent } from "@pipecat-ai/client-react";
import {
  DataList,
  PanelContent,
  PanelHeader,
  PanelTitle,
} from "@pipecat-ai/voice-ui-kit";

type StageMetric = {
  processor: string;
  ms: number;
};

const formatMs = (value?: number | null) =>
  typeof value === "number" ? `${Math.round(value)} ms` : "—";

const formatTimestamp = (value: Date | null) =>
  value ? value.toLocaleTimeString() : "—";

const mapMetrics = (metrics?: PipecatMetricData[]) =>
  (metrics ?? []).map((metric) => ({
    processor: metric.processor,
    ms: metric.value * 1000,
  }));

const sumMetrics = (metrics: StageMetric[]) =>
  metrics.reduce((total, metric) => total + metric.ms, 0);

const buildMetricData = (metrics: StageMetric[]) =>
  metrics.reduce<Record<string, string>>((acc, metric) => {
    acc[metric.processor] = formatMs(metric.ms);
    return acc;
  }, {});

function LatencyMetricsSection() {
  const [speechToAudioMs, setSpeechToAudioMs] = useState<number | null>(null);
  const [ttfbMetrics, setTtfbMetrics] = useState<StageMetric[]>([]);
  const [processingMetrics, setProcessingMetrics] = useState<StageMetric[]>([]);
  const [lastMetricsAt, setLastMetricsAt] = useState<Date | null>(null);
  const lastUserStopAt = useRef<number | null>(null);

  useRTVIClientEvent(RTVIEvent.Connected, () => {
    setSpeechToAudioMs(null);
    setTtfbMetrics([]);
    setProcessingMetrics([]);
    setLastMetricsAt(null);
    lastUserStopAt.current = null;
  });

  useRTVIClientEvent(RTVIEvent.UserStoppedSpeaking, () => {
    lastUserStopAt.current = performance.now();
  });

  useRTVIClientEvent(RTVIEvent.BotStartedSpeaking, () => {
    if (lastUserStopAt.current === null) return;
    setSpeechToAudioMs(performance.now() - lastUserStopAt.current);
  });

  useRTVIClientEvent(RTVIEvent.Metrics, (data) => {
    const nextTtfb = mapMetrics(data?.ttfb);
    const nextProcessing = mapMetrics(data?.processing);
    if (nextTtfb.length > 0) setTtfbMetrics(nextTtfb);
    if (nextProcessing.length > 0) setProcessingMetrics(nextProcessing);
    if (nextTtfb.length > 0 || nextProcessing.length > 0) {
      setLastMetricsAt(new Date());
    }
  });

  const totalTtfbMs = useMemo(() => sumMetrics(ttfbMetrics), [ttfbMetrics]);
  const totalProcessingMs = useMemo(
    () => sumMetrics(processingMetrics),
    [processingMetrics]
  );

  const summaryData = useMemo(
    () => ({
      "Speech → audio": formatMs(speechToAudioMs),
      "Total TTFB (sum)": ttfbMetrics.length ? formatMs(totalTtfbMs) : "—",
      "Total processing (sum)": processingMetrics.length
        ? formatMs(totalProcessingMs)
        : "—",
      "Last update": formatTimestamp(lastMetricsAt),
    }),
    [
      speechToAudioMs,
      ttfbMetrics.length,
      totalTtfbMs,
      processingMetrics.length,
      totalProcessingMs,
      lastMetricsAt,
    ]
  );

  const ttfbData = useMemo(() => buildMetricData(ttfbMetrics), [ttfbMetrics]);
  const processingData = useMemo(
    () => buildMetricData(processingMetrics),
    [processingMetrics]
  );

  const hasMetrics =
    speechToAudioMs !== null ||
    ttfbMetrics.length > 0 ||
    processingMetrics.length > 0;

  return (
    <>
      <PanelHeader className="border-t border-t-border" variant="inline">
        <PanelTitle>Latency</PanelTitle>
      </PanelHeader>
      <PanelContent>
        {!hasMetrics && (
          <div className="text-sm text-muted-foreground">
            No latency data yet. Start a session to populate metrics.
          </div>
        )}
        {hasMetrics && (
          <div className="flex flex-col gap-3 text-sm">
            <DataList data={summaryData} />
            {ttfbMetrics.length > 0 && (
              <div>
                <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">
                  TTFB per stage
                </div>
                <DataList data={ttfbData} />
              </div>
            )}
            {processingMetrics.length > 0 && (
              <div>
                <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">
                  Processing time per stage
                </div>
                <DataList data={processingData} />
              </div>
            )}
          </div>
        )}
      </PanelContent>
    </>
  );
}

export default function LatencyMetricsPortal() {
  const [portalTarget, setPortalTarget] = useState<HTMLElement | null>(null);
  const portalNodeRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (typeof document === "undefined") return;

    const attachPortal = () => {
      if (portalNodeRef.current) return;
      const infoPanel = document.getElementById("info-panel");
      const panelRoot = infoPanel?.querySelector('[data-slot="panel"]');
      if (!panelRoot) return;
      const existing = panelRoot.querySelector<HTMLElement>(
        '[data-latency-metrics="true"]'
      );
      if (existing) {
        portalNodeRef.current = existing;
        setPortalTarget(existing);
        return;
      }
      const container = document.createElement("div");
      container.dataset.latencyMetrics = "true";
      panelRoot.appendChild(container);
      portalNodeRef.current = container;
      setPortalTarget(container);
    };

    const ensurePortal = () => {
      if (
        portalNodeRef.current &&
        !document.body.contains(portalNodeRef.current)
      ) {
        portalNodeRef.current = null;
        setPortalTarget(null);
      }
      attachPortal();
    };

    ensurePortal();
    const observer = new MutationObserver(ensurePortal);
    observer.observe(document.body, { childList: true, subtree: true });
    return () => {
      observer.disconnect();
      portalNodeRef.current?.remove();
      portalNodeRef.current = null;
      setPortalTarget(null);
    };
  }, []);

  if (!portalTarget) return null;
  return createPortal(<LatencyMetricsSection />, portalTarget);
}
