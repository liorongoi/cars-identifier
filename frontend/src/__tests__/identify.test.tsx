import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { IdentifyPage } from "../pages/IdentifyPage";
import { RESULT } from "./fixtures";

const ok = (body: unknown) => Promise.resolve(new Response(JSON.stringify(body), { status: 200 }));

describe("IdentifyPage", () => {
  beforeEach(() => {
    URL.createObjectURL = vi.fn(() => "blob:preview");
    URL.revokeObjectURL = vi.fn();
  });
  afterEach(() => vi.restoreAllMocks());

  it("uploads an image and shows the ranked result", async () => {
    const fetchMock = vi.fn((url: string, init?: RequestInit) => {
      if (url === "/api/catalog") return ok({ total_images: 0, makes: [] });
      if (url === "/api/identify") {
        expect(init?.method).toBe("POST");
        expect((init?.body as FormData).get("top_k")).toBe("10");
        return ok(RESULT);
      }
      return Promise.reject(new Error(`unexpected ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<MemoryRouter><IdentifyPage /></MemoryRouter>);
    const button = screen.getByRole("button", { name: /Identify/ });
    expect(button).toBeDisabled();
    await userEvent.upload(screen.getByTestId("file-input"), new File(["x"], "dash.jpg", { type: "image/jpeg" }));
    expect(button).toBeEnabled();
    await userEvent.click(button);
    await waitFor(() => expect(screen.getByText("Ranked candidates (2)")).toBeTruthy());
    expect(screen.getAllByText("Toyota Corolla").length).toBeGreaterThan(0);
  });

  it("shows API errors", async () => {
    vi.stubGlobal("fetch", vi.fn((url: string) =>
      url === "/api/catalog"
        ? ok({ total_images: 0, makes: [] })
        : Promise.resolve(new Response(JSON.stringify({ detail: "not a recognised image file" }), { status: 422 })),
    ));
    render(<MemoryRouter><IdentifyPage /></MemoryRouter>);
    await userEvent.upload(screen.getByTestId("file-input"), new File(["x"], "a.jpg", { type: "image/jpeg" }));
    await userEvent.click(screen.getByRole("button", { name: /Identify/ }));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("not a recognised image file"));
  });

  it("rejects non-image files", async () => {
    vi.stubGlobal("fetch", vi.fn(() => ok({ total_images: 0, makes: [] })));
    render(<MemoryRouter><IdentifyPage /></MemoryRouter>);
    await userEvent.upload(screen.getByTestId("file-input"), new File(["x"], "notes.txt", { type: "text/plain" }),
      { applyAccept: false });
    expect(screen.getByRole("alert")).toHaveTextContent("is not an image");
  });
});
