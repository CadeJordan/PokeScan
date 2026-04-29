const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000/api";

export type CenteringSubGrade = {
  grade: number | null;
  left_right: string | null;
  top_bottom: string | null;
};

export type SubGrades = {
  centering: CenteringSubGrade | null;
  corners: number | null;
  edges: number | null;
  surface: number | null;
};

export type GradeResponse = {
  grade: number;
  grade_int: number;
  confidence: number;
  rank_probs: number[];
  sub_grades: SubGrades;
  notes: string[];
};

export type HealthResponse = {
  status: string;
  model_loaded: boolean;
  metadata: Record<string, unknown>;
};

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error(`health check failed (${res.status})`);
  return res.json();
}

export async function gradeCard(front: File, back: File): Promise<GradeResponse> {
  const form = new FormData();
  form.append("front", front);
  form.append("back", back);
  const res = await fetch(`${API_BASE}/grade`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res
      .json()
      .then((b) => (b as { detail?: string }).detail)
      .catch(() => undefined);
    throw new Error(detail ?? `grade request failed (${res.status})`);
  }
  return res.json();
}
