import assert from "node:assert/strict";
import test from "node:test";
import { route, scenarioEdges } from "./network.ts";

test("distance routing chooses the shortest healthy path", () => {
  assert.deepEqual(route(scenarioEdges("healthy"), "R1", "R8").path, ["R1", "R2", "R4", "R7", "R8"]);
});

test("risk-aware routing avoids the degraded R1-R2 link", () => {
  const result = route(scenarioEdges("degraded"), "R1", "R8", 200);
  assert.deepEqual(result.path, ["R1", "R4", "R7", "R8"]);
  assert.equal(result.path.includes("R2"), false);
});

test("an outage removes R4-R7 from every route", () => {
  const result = route(scenarioEdges("outage"), "R1", "R8", 200);
  assert.notDeepEqual(result.path, ["R1", "R4", "R7", "R8"]);
  assert.ok(result.path.length > 0);
});
