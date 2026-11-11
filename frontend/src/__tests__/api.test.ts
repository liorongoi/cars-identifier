import { describe, expect, it } from "vitest";
import { buildIdentifyForm, imageUrl, pct } from "../api";
import { parseModelNames } from "../pages/AddMakePage";

describe("api helpers", () => {
  it("builds the identify form with only the set options", () => {
    const file = new File(["x"], "q.jpg", { type: "image/jpeg" });
    const fd = buildIdentifyForm({ file, topK: 7, enhance: true, maskFaces: false, makeIds: [1, 2],
      manualParts: [{ part: "steering_wheel", bbox: [1, 2, 3, 4] }], caseRef: "C-1" });
    expect(fd.get("top_k")).toBe("7");
    expect(fd.get("mask_faces")).toBe("false");
    expect(fd.get("make_ids")).toBe("1,2");
    expect(JSON.parse(String(fd.get("manual_parts")))[0].part).toBe("steering_wheel");
    expect(fd.get("case_ref")).toBe("C-1");
    expect(fd.has("mode")).toBe(false);
    expect(fd.has("year_min")).toBe(false);
    expect(fd.has("analyst")).toBe(false);
  });

  it("formats percentages", () => {
    expect(pct(0.7234)).toBe("72.3%");
  });

  it("builds image urls", () => {
    expect(imageUrl(5)).toBe("/api/images/5/file");
    expect(imageUrl(5, { size: 200, bbox: [1, 2, 3, 4] })).toBe("/api/images/5/file?size=200&bbox=1%2C2%2C3%2C4");
  });

  it("parses model name lists", () => {
    expect(parseModelNames("Tiggo 7, Tiggo 8\nArrizo 5; tiggo 7")).toEqual(["Tiggo 7", "Tiggo 8", "Arrizo 5"]);
    expect(parseModelNames("")).toEqual([]);
  });
});
