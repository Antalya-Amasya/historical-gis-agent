export type DemoCampaign = {
  id: "hannibal" | "caesar";
  label: string;
  prompt: string;
  presentationRouteId?: string;
};

export const DEMO_CAMPAIGNS: DemoCampaign[] = [
  {
    id: "hannibal",
    label: "Hannibal’s invasion of Italy (218 BCE)",
    prompt: "展示汉尼拔翻越阿尔卑斯进入意大利的路线",
  },
  {
    id: "caesar",
    label: "Caesar conquest of Gaul",
    prompt: "展示凯撒征服高卢路线",
    presentationRouteId: "caesar-gallic-campaign",
  },
];

export const DEMO_EVALUATION = {
  corpora: ["Polybius, Histories", "Livy, Ab Urbe Condita", "Caesar, De Bello Gallico"],
  pipeline: ["Question", "Intent", "Evidence", "Historical anchors", "Geographic constraints", "A*", "GeoJSON"],
  verification: ["Backend tests: 215 passed", "Geography MCP: 3 passed"],
};
