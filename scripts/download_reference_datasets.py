#!/usr/bin/env python3
"""
Automated downloader for open laboratory and diagnostic reference datasets.

This script focuses on the openly accessible resources enumerated in labs_guide.md.
Datasets that require authentication, licensing, or manual FOI requests are listed
as manual follow-ups and are not downloaded automatically.

Usage examples:
    python scripts/download_reference_datasets.py
    python scripts/download_reference_datasets.py --dry-run
    python scripts/download_reference_datasets.py --force
    python scripts/download_reference_datasets.py --list-manual
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import shutil
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_ROOT = Path("outputs/reference_datasets")


@dataclass(frozen=True)
class Resource:
    """Definition of a downloadable resource."""

    name: str
    url: str
    relative_path: Path
    description: str


@dataclass(frozen=True)
class ManualResource:
    """Definition of a resource that requires manual acquisition."""

    name: str
    link: str
    notes: str


AUTO_RESOURCES: tuple[Resource, ...] = (
    Resource(
        name="HL7 US Core FHIR package",
        url="https://build.fhir.org/ig/HL7/US-Core/package.tgz",
        relative_path=Path("fhir/us-core/package.tgz"),
        description="FHIR Implementation Guide package that includes ValueSet-us-core-observation-lab-codes.json.",
    ),
    Resource(
        name="LABO ontology",
        url="http://purl.obolibrary.org/obo/labo.owl",
        relative_path=Path("ontologies/labo.owl"),
        description="Clinical Laboratory Ontology (OBO) for lab informational entities.",
    ),
    Resource(
        name="NCBI new_taxdump",
        url="https://ftp.ncbi.nih.gov/pub/taxonomy/new_taxdump/new_taxdump.zip",
        relative_path=Path("microbiology/new_taxdump.zip"),
        description="NCBI taxonomy snapshot for organism identifiers in microbiology mapping.",
    ),
    Resource(
        name="DICOM PS3.16 HTML",
        url="https://dicom.nema.org/medical/dicom/current/output/chtml/part16/PS3.16.html",
        relative_path=Path("imaging/dicom_ps3.16.html"),
        description="DICOM Part 16 controlled terminology tables (HTML snapshot).",
    ),
    Resource(
        name="HL7 specimen codes (v2-0487)",
        url="https://terminology.hl7.org/5.5.0/CodeSystem-v2-0487.json",
        relative_path=Path("specimens/CodeSystem-v2-0487.json"),
        description="HL7 v2 Table 0487 specimen type codes (JSON).",
    ),
    Resource(
        name="HL7 specimen codes (v2-0070)",
        url="https://terminology.hl7.org/5.5.0/CodeSystem-v2-0070.json",
        relative_path=Path("specimens/CodeSystem-v2-0070.json"),
        description="HL7 v2 Table 0070 specimen source codes (JSON).",
    ),
    Resource(
        name="UCUM repository (GitHub ZIP)",
        url="https://github.com/ucum-org/ucum/archive/refs/heads/main.zip",
        relative_path=Path("units/ucum-main.zip"),
        description="UCUM specification and XML essence (GitHub repository archive).",
    ),
    Resource(
        name="QUDT public repository (GitHub ZIP)",
        url="https://github.com/qudt/qudt-public-repo/archive/refs/heads/main.zip",
        relative_path=Path("units/qudt-public-repo.zip"),
        description="QUDT vocabularies for quantity kinds and unit semantics.",
    ),
    Resource(
        name="Uberon anatomy ontology",
        url="http://purl.obolibrary.org/obo/uberon.owl",
        relative_path=Path("ontologies/uberon.owl"),
        description="Uberon anatomical ontology (OBO).",
    ),
    Resource(
        name="OBI ontology",
        url="http://purl.obolibrary.org/obo/obi.owl",
        relative_path=Path("ontologies/obi.owl"),
        description="Ontology for Biomedical Investigations, covering assay methods.",
    ),
    Resource(
        name="HL7 FHIR ImagingStudy specification",
        url="https://hl7.org/fhir/imagingstudy.html",
        relative_path=Path("imaging/fhir_imagingstudy.html"),
        description="FHIR resource definition for ImagingStudy (HTML snapshot).",
    ),
    Resource(
        name="OHDSI Vocabulary GitHub helper",
        url="https://github.com/OHDSI/Vocabulary-v5.0/archive/refs/heads/master.zip",
        relative_path=Path("omop/ohdsi-vocabulary-master.zip"),
        description="Public OHDSI vocabulary repository (contains LOINC-SNOMED helper SQL).",
    ),
)

MANUAL_RESOURCES: tuple[ManualResource, ...] = (
    ManualResource(
        name="LOINC master terminology package",
        link="https://loinc.org/downloads/",
        notes="Requires free LOINC account and license acceptance before downloading the ZIP release.",
    ),
    ManualResource(
        name="SNOMED CT International Edition",
        link="https://www.snomed.org/snomed-ct/get-snomed-ct",
        notes="Accessible only with SNOMED International or UMLS credentials; redistribution restricted.",
    ),
    ManualResource(
        name="OMOP Vocabulary (Athena)",
        link="https://athena.ohdsi.org/",
        notes="Register for an OHDSI Athena account, request vocabularies, and download the emailed ZIP.",
    ),
    ManualResource(
        name="WHO Essential Diagnostics List",
        link="https://www.who.int/teams/health-product-and-policy-standards/standards-and-specifications/essential-diagnostics-list",
        notes="Download current PDF/Excel manually from WHO site; capture version and publication date.",
    ),
    ManualResource(
        name="UNC i2b2 Common Lab Panels",
        link="https://tracs.unc.edu/index.php/services/biomedical-informatics/i2b2",
        notes="Documentation may require requesting access from UNC TRACS; archive any received panel lists.",
    ),
    ManualResource(
        name="NHS National Laboratory Medicine Catalogue (NLMC)",
        link="https://isd.digital.nhs.uk/trud3/user/guest/group/0/pack/26",
        notes="Create NHS TRUD account, accept NLMC license, and download the ZIP manually.",
    ),
    ManualResource(
        name="Philippine institutional lab catalogues",
        link="https://www.pgh.gov.ph/",
        notes="Collect published price lists or request via FOI portals; store PDFs under version control.",
    ),
    ManualResource(
        name="PhilHealth benefit package circulars",
        link="https://www.philhealth.gov.ph/partners/providers/payments/case_rates.html",
        notes="Download relevant circular PDFs that enumerate diagnostic inclusions.",
    ),
    ManualResource(
        name="WHONET antimicrobial code list",
        link="https://whonet.org/",
        notes="Retrieve from Documents > Antibiotic codes; site blocks automated fetches so download manually.",
    ),
    ManualResource(
        name="ATC/DDD index exports (antibacterials J01)",
        link="https://www.whocc.no/atc_ddd_index/",
        notes="Use web export tools to download CSV/Excel for required ATC sections.",
    ),
    ManualResource(
        name="RadLex Playbook (LOINC-RSNA)",
        link="https://loinc.org/download/radlex-playbook/",
        notes="Requires LOINC login and RSNA terms acceptance before downloading the CSV.",
    ),
    ManualResource(
        name="WHO International Classification of Health Interventions (ICHI)",
        link="https://www.who.int/standards/classifications/international-classification-of-health-interventions",
        notes="Download Excel/PDF from WHO page; some components may need email approval.",
    ),
    ManualResource(
        name="CPT and HCPCS procedure codes",
        link="https://www.ama-assn.org/practice-management/cpt",
        notes="CPT requires paid license; HCPCS Level II ZIPs are available from CMS but need manual download.",
    ),
    ManualResource(
        name="UMLS Metathesaurus",
        link="https://uts.nlm.nih.gov/uts/",
        notes="Sign the annual UMLS license and use the download API or browser to fetch the Metathesaurus bundle.",
    ),
    ManualResource(
        name="DOH Philippines clinical laboratory regulations",
        link="https://doh.gov.ph/clinical-laboratory",
        notes="Download Administrative Orders and annexes (some require FOI requests); archive PDFs.",
    ),
    ManualResource(
        name="Local terminology glossary",
        link="Internal curation",
        notes="Maintain synonyms and abbreviations gathered from stakeholders and local documentation.",
    ),
)


def download_resource(resource: Resource, root: Path, force: bool, dry_run: bool) -> Optional[dict]:
    """
    Download a single resource if needed.

    Returns a manifest entry (dict) on success, None otherwise.
    """
    target = (root / resource.relative_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not force:
        print(f"[SKIP] {resource.name} already exists at {target}")
        return None

    if dry_run:
        print(f"[DRY] Would download {resource.name} -> {target}")
        return None

    tmp_path = target.with_suffix(target.suffix + ".partial")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"[GET] {resource.name}")
    try:
        with urllib.request.urlopen(resource.url) as response, open(tmp_path, "wb") as fh:
            shutil.copyfileobj(response, fh)
    except urllib.error.URLError as exc:
        print(f"[FAIL] {resource.name}: {exc}", file=sys.stderr)
        if tmp_path.exists():
            tmp_path.unlink()
        return None

    tmp_path.rename(target)
    sha256 = compute_sha256(target)
    size_bytes = target.stat().st_size
    timestamp = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    print(f"[OK] {resource.name} -> {target} ({size_bytes:,} bytes, sha256={sha256[:12]}...)")

    return {
        "name": resource.name,
        "url": resource.url,
        "path": str(target.relative_to(root)),
        "description": resource.description,
        "bytes": size_bytes,
        "sha256": sha256,
        "downloaded_at": timestamp,
    }


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate the SHA-256 digest for a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def list_manual_resources(resources: Iterable[ManualResource]) -> None:
    """Pretty-print the manual resource checklist."""
    print("Manual acquisition required for the following datasets:\n")
    for item in resources:
        wrapped_notes = textwrap.fill(item.notes, width=88)
        print(f"- {item.name}")
        print(f"  Link : {item.link}")
        print(f"  Notes: {wrapped_notes}\n")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download open laboratory and diagnostic reference datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Output directory where datasets will be stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be downloaded without fetching them.",
    )
    parser.add_argument(
        "--list-manual",
        action="store_true",
        help="List datasets that still require manual acquisition, then exit.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    if args.list_manual:
        list_manual_resources(MANUAL_RESOURCES)
        return 0

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict] = []
    for resource in AUTO_RESOURCES:
        entry = download_resource(resource, root=root, force=args.force, dry_run=args.dry_run)
        if entry:
            manifest_entries.append(entry)

    if args.dry_run:
        print("\nDry run complete. No files were downloaded.")
        return 0

    if manifest_entries:
        manifest_path = root / "manifest.json"
        manifest = {
            "generated_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "root": str(root),
            "resources": manifest_entries,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"\nManifest written to {manifest_path}")
    else:
        print("\nNo downloads were performed (all files already present).")

    print("\nReminder: run with --list-manual to review datasets that still require manual acquisition.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
