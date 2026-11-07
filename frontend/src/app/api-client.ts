import { AccessCodeError, accessCodeHeaders } from "@/lib/accessCode";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000/api";

export type CenteringSubGrade = {
  grade: number | null;
  left_right: string | null;
  top_bottom: string | null;
  back_left_right: string | null;
  back_top_bottom: string | null;
};

export type FactorSubGrade = {
  grade: number | null;
  hint: string | null;
  flags: string[];
};

export type SubGrades = {
  centering: CenteringSubGrade | null;
  corners: FactorSubGrade | null;
  edges: FactorSubGrade | null;
  surface: FactorSubGrade | null;
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

// Throws AccessCodeError on a 401 (the API's ACCESS_CODE is set and `accessCode`
// is missing or wrong); any other failure is an Error with the API's detail.
export async function gradeCard(
  front: File,
  back: File,
  accessCode: string | null = null,
): Promise<GradeResponse> {
  const form = new FormData();
  form.append("front", front);
  form.append("back", back);
  const res = await fetch(`${API_BASE}/grade`, {
    method: "POST",
    body: form,
    headers: accessCodeHeaders(accessCode),
  });
  if (!res.ok) {
    const detail = await res
      .json()
      .then((b) => (b as { detail?: string }).detail)
      .catch(() => undefined);
    if (res.status === 401) throw new AccessCodeError(detail);
    throw new Error(detail ?? `grade request failed (${res.status})`);
  }
  return res.json();
}
