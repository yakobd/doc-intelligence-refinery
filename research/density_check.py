from pathlib import Path

import pdfplumber
from tabulate import tabulate


PDF_FILES = [
	"data/raw/CBE Annual Report 2015-16_1.pdf",
	"data/raw/Consumer Price Index March 2025.pdf",
	"data/raw/Security_Vulnerability_Disclosure_Standard_Procedure_2.pdf",
]


def likely_stamp_or_handwriting(images: list[dict], page_area: float) -> tuple[bool, int]:
	"""Heuristic: many small image objects may indicate stamps/handwriting scans."""
	if not images or page_area <= 0:
		return False, 0

	small_images = []
	for image in images:
		width = float(image.get("width", 0) or 0)
		height = float(image.get("height", 0) or 0)
		area = width * height
		if area > 0 and area <= page_area * 0.01:
			small_images.append(image)

	small_count = len(small_images)
	return small_count >= 5, small_count


def analyze_first_page(pdf_path: Path) -> dict:
	if not pdf_path.exists():
		return {
			"file": str(pdf_path),
			"status": "missing",
			"total_chars": None,
			"image_objects": None,
			"char_density": None,
			"small_images": None,
			"possible_stamps_or_handwriting": None,
		}

	try:
		with pdfplumber.open(pdf_path) as pdf:
			if not pdf.pages:
				return {
					"file": str(pdf_path),
					"status": "no_pages",
					"total_chars": 0,
					"image_objects": 0,
					"char_density": 0.0,
					"small_images": 0,
					"possible_stamps_or_handwriting": "No",
				}

			page = pdf.pages[0]
			text = page.extract_text() or ""
			total_chars = len(text)

			images = page.images or []
			image_objects = len(images)

			page_area = float(page.width) * float(page.height)
			char_density = (total_chars / page_area) if page_area > 0 else 0.0

			maybe_present, small_count = likely_stamp_or_handwriting(images, page_area)

			return {
				"file": str(pdf_path),
				"status": "ok",
				"total_chars": total_chars,
				"image_objects": image_objects,
				"char_density": char_density,
				"small_images": small_count,
				"possible_stamps_or_handwriting": "Yes" if maybe_present else "No",
			}
	except Exception as exc:
		return {
			"file": str(pdf_path),
			"status": f"error: {exc}",
			"total_chars": None,
			"image_objects": None,
			"char_density": None,
			"small_images": None,
			"possible_stamps_or_handwriting": None,
		}


def main() -> None:
	project_root = Path(__file__).resolve().parents[1]
	results = [analyze_first_page(project_root / relative_path) for relative_path in PDF_FILES]

	table = []
	for item in results:
		table.append(
			[
				Path(item["file"]).name,
				item["status"],
				item["total_chars"],
				item["image_objects"],
				f"{item['char_density']:.8f}" if isinstance(item["char_density"], float) else item["char_density"],
				item["small_images"],
				item["possible_stamps_or_handwriting"],
			]
		)

	headers = [
		"File",
		"Status",
		"Total Chars",
		"Image Objects",
		"Char Density",
		"Small Images",
		"Stamps/Handwriting?",
	]

	print(tabulate(table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
	main()
