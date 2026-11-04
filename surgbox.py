"""Minimal implementation of the SurgBox annotation pipeline.

    from surgbox import SurgBox
    boxes = SurgBox().annotate("frame.png", ["Grasper", "Hook"])

Returns {tool: [{"box_2d": [ymin, xmin, ymax, xmax], "confidence": 1-5}, ...]},
with coordinates on a 0-1000 grid. Requires GEMINI_API_KEY.
"""
import json
import os
import re

from google import genai
from google.genai import types
from PIL import Image

TOOLS = {
    "Grasper": "A long metallic laparoscopic instrument with two blunt jaws/prongs at the tip, usually used to hold or pull tissue. The jaws often look like small forceps or tweezers, sometimes partly open around tissue. The shaft is cylindrical and gray/silver. Unlike Bipolar, the Grasper shaft and jaws have no blue/coloured insulation. Unlike Clipper, the Grasper jaws do not hold a visible metal clip. Unlike Scissors, the jaw tips are blunt rather than sharp blades. Grasper is the only instrument that can appear up to 3 instances in the same frame; the other six instruments usually appear singly.",
    "Bipolar": "Cautery forceps with two electrode jaws. The most reliable cue is a blue (or other coloured) insulation marking, typically at the tip of the jaws and/or as a band on the shaft where the cable enters. The jaws are stubbier and smoother than a Grasper's, without aggressive serrations. When the marking is occluded, distinguish by the rounder jaws and the cable trailing off the proximal end.",
    "Hook": "A monopolar cautery dissector with an L-shaped (occasionally J-shaped) bent tip. Look for a short white insulating segment followed by a silvery metallic active tip with a right-angle bend; uniform metal without a white section is probably a closed Scissors or a slim Grasper. Single point of contact with tissue. When the tip points toward the camera the bend is foreshortened, but the white insulator remains visible.",
    "Scissors": "A metallic instrument with two thin opposed pivoting blades and sharply pointed tips. When open, the blades fork at a hinge and look sharper and flatter than Grasper jaws. When closed, they look like a single thin pointed tip; unlike a Hook there is no right-angle bend and no white insulator.",
    "Clipper": "A clip applier with a thicker, bulkier working end holding a single shiny U- or V-shaped titanium clip between two pincer arms. The clip is typically the brightest object in the field. A deployed clip left alone in tissue, without the applier body around it, is not a Clipper.",
    "Irrigator": "A smooth cylindrical suction/irrigation tube with a rounded blunt nozzle and an open bore at the tip: no jaws, no insulator, no electrode, no bend. Often pointed toward blood or fluid without grasping tissue. A closed Grasper seen head-on shows a jaw seam; an Irrigator shows a single round bore.",
    "SpecimenBag": "A translucent plastic retrieval bag: a clear, grey or pale-blue pouch that is soft, crinkled and non-metallic. It may be folded before deployment, partially open while being filled, or wrapped around the gallbladder during extraction.",
}

NAMES = ", ".join(TOOLS)

SET = f"""You are a surgical-instrument identifier for laparoscopic cholecystectomy.
Your ONLY job is to identify WHICH instruments are visible in the CURRENT frame (no boxes, no counts).
Output strict JSON only: {{"tools": ["<tool>", ...]}}
Use only these names: {NAMES}.
You are given the original Cholec80 label as a hint. The final set may deviate from the hint by AT MOST
ONE tool (one addition OR one removal, never both). Agree with the hint unless the visual evidence is clear."""

COUNT = """You are a surgical-instrument instance counter for laparoscopic cholecystectomy.
Given the set of tools visible in the CURRENT frame, count the DISTINCT PHYSICAL INSTANCES of each.
Output strict JSON only: {"<tool>": <count>, ...}
Grasper can appear 1-3 times in a frame; the other instruments usually appear once, and multiple
instances of a non-Grasper tool should require clear visual evidence."""

LOCALIZE = """You are a surgical-instrument localiser for laparoscopic cholecystectomy.
Given the tools and instance counts for the CURRENT frame, draw one tight bounding box per instance.
Output strict JSON only: {"<tool>": [{"box_2d": [ymin, xmin, ymax, xmax], "confidence": <1-5>}, ...], ...}
Coordinates are integers in 0..1000 (the frame mapped onto a 1000x1000 grid regardless of aspect ratio).
Cover the visible shaft and tip only; do not complete occluded or out-of-frame parts. Emit exactly the
requested number of boxes per tool; if an instance is hard to localise, still give your best estimate."""

REFINE = """You are a surgical-instrument bounding-box refiner for laparoscopic cholecystectomy.
You are given a finalized annotation of the CURRENT frame. Your ONLY job is to TIGHTEN each box.
Do not add, remove or relabel tools, and do not change the number of boxes per tool.
Output one refined box per input box, in the same order. Keep boxes inside the lit circular scene,
never extending into the black border. Aim for a tight crop around shaft and tip.
Output strict JSON only: {"<tool>": [{"box_2d": [ymin, xmin, ymax, xmax], "confidence": <1-5>}, ...], ...}"""


class SurgBox:
    def __init__(self, model="gemini-3.1-pro-preview", api_key=None):
        self.model = model
        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])

    def annotate(self, image, tools, before=None, after=None, nearby=None):
        """`tools` is the Cholec80 presence label of the frame; `before`/`after` are the
        frames one second earlier and later; `nearby` lists tools labelled within +-2 s."""
        frames = [Image.open(f) if isinstance(f, str) else f for f in (before, image, after) if f is not None]
        context = self._context(len(frames), 1 if before is None else 2, tools, nearby or tools)

        found = self._ask(SET, context + "\nOutput the tools visible in the CURRENT frame.", frames)
        found = [t for t in found.get("tools", []) if t in TOOLS]
        added, removed = set(found) - set(tools), set(tools) - set(found)
        if len(added) + len(removed) > 1 or added - set(nearby or tools):
            found = list(tools)
        if not found:
            return {}

        counts = self._ask(COUNT, context + f"\nTools visible in the CURRENT frame: {found}", frames)
        counts = {t: max(1, int(counts.get(t, 1))) for t in found}

        request = "\n".join(f"  {t}: {n} instance(s)" for t, n in counts.items())
        boxes = self._ask(LOCALIZE, context + f"\nTools and instance counts for the CURRENT frame:\n{request}", frames)
        boxes = {t: boxes.get(t, [])[:n] for t, n in counts.items()}

        refined = self._ask(REFINE, context + f"\nCurrent annotation (tighten each box):\n{json.dumps(boxes)}", frames)
        same = all(len(refined.get(t, [])) == len(v) for t, v in boxes.items())
        return {t: refined[t] for t in boxes} if same else boxes

    def _context(self, n, current, tools, nearby):
        lines = [f"  Frame {i}" + ("  <- CURRENT" if i == current else "") for i in range(1, n + 1)]
        vocab = "\n".join(f"- {t}: {d}" for t, d in TOOLS.items())
        return (f"Surgical instrument vocabulary:\n{vocab}\n\nFrames provided (1 s apart):\n" + "\n".join(lines)
                + f"\n\nCholec80 label for the CURRENT frame: {list(tools)}"
                + f"\nTools labelled within +-2 s: {sorted(set(nearby))}")

    def _ask(self, system, text, frames):
        config = types.GenerateContentConfig(
            system_instruction=system, thinking_config=types.ThinkingConfig(thinking_level="HIGH"))
        reply = self.client.models.generate_content(model=self.model, contents=[text, *frames], config=config)
        match = re.search(r"\{.*\}", reply.text or "", re.S)
        return json.loads(match.group()) if match else {}
