import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { ResultsTable } from "../components/ResultsTable";
import { ResultView } from "../components/ResultView";
import { RESULT } from "./fixtures";

describe("ResultsTable", () => {
  it("shows ranked candidates with probabilities and reference counts", () => {
    render(<ResultsTable candidates={RESULT.candidates} queryParts={RESULT.parts} imageSize={[800, 600]} queryImg={null} />);
    const rows = screen.getAllByRole("row");
    expect(rows[1]).toHaveTextContent("Toyota Corolla");
    expect(rows[1]).toHaveTextContent("72.0%");
    expect(rows[1]).toHaveTextContent("48");
    expect(rows[1]).toHaveTextContent("2018–present");
    expect(screen.getByText("Honda Civic")).toBeTruthy();
    // first candidate's evidence is open by default
    expect(screen.getByAltText("reference steering_wheel")).toHaveAttribute(
      "src", "/api/images/55/file?size=400&bbox=1%2C2%2C30%2C40");
  });

  it("toggles evidence rows", async () => {
    render(<ResultsTable candidates={RESULT.candidates} queryParts={RESULT.parts} imageSize={[800, 600]} queryImg={null} />);
    await userEvent.click(screen.getByText("Toyota Corolla"));
    expect(screen.queryByAltText("reference steering_wheel")).toBeNull();
  });

  it("renders the empty state", () => {
    render(<ResultsTable candidates={[]} queryParts={[]} imageSize={[1, 1]} queryImg={null} />);
    expect(screen.getByText("No candidates.")).toBeTruthy();
  });
});

describe("ResultView", () => {
  it("shows best match, warnings, quality and exports", () => {
    render(<MemoryRouter><ResultView result={RESULT} imageSrc="blob:x" /></MemoryRouter>);
    expect(screen.getByText(/Poor image quality/)).toBeTruthy();
    expect(screen.getByText("high confidence")).toBeTruthy();
    expect(screen.getByText(/fair · 73\/100/)).toBeTruthy();
    expect(screen.getByText("1 face(s) masked")).toBeTruthy();
    expect(screen.getByText("Export CSV")).toHaveAttribute("href", "/api/identifications/7/export?format=csv");
  });
});
