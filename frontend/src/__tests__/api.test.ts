import { describe, expect, it } from "vitest";
import { buildIdentifyForm, formatYears, imageUrl, pct } from "../api";
import { parseGenerations } from "../pages/AddMakePage";

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

  it("formats years and percentages", () => {
    expect(formatYears(2018, null)).toBe("2018–present");
    expect(formatYears(2015, 2015)).toBe("2015");
    expect(formatYears(2010, 2014)).toBe("2010–2014");
    expect(formatYears(null, null)).toBe("years unknown");
    expect(pct(0.7234)).toBe("72.3%");
  });

  it("builds image urls", () => {
    expect(imageUrl(5)).toBe("/api/images/5/file");
    expect(imageUrl(5, { size: 200, bbox: [1, 2, 3, 4] })).toBe("/api/images/5/file?size=200&bbox=1%2C2%2C3%2C4");
  });

  it("parses generation shorthand", () => {
    expect(parseGenerations("E170:2013-2019; E210:2018-")).toEqual([
      { name: "E170", year_start: 2013, year_end: 2019 },
      { name: "E210", year_start: 2018, year_end: null },
    ]);
    expect(parseGenerations("")).toEqual([]);
    expect(() => parseGenerations("bad")).toThrow(/NAME:START-END/);
  });
});
