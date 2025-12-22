"""Static constants for CSS selectors and mappings.

These values should NOT be configurable via environment variables
as they are tightly coupled to website structure.
"""

# CSS Selectors for Playwright (HikVision)
HIKVISION_SELECTORS = {
    "search_list": ".search-list",
    "subcategory_dropdown": "[data-title-type='subcategory']",
    "subcategory_radio": "input[type='radio'][value='{subcategory}']",
    "product_count": ".sum-number-of-products",
    "product_grid": ".layout4-wrapper",
    "product_link": ".btn-details-link",
    "next_page_btn": "li span.next",
    "pagination": ".pagination-section",
}

# Subcategory filter values (exact text as shown in UI)
HIKVISION_IP_SUBCATEGORIES = {
    "Network Cameras": "Network Cameras",
    "PTZ Cameras": "PTZ Cameras",
    "Explosion-Proof Series": "Explosion-Proof and Anti-Corrosion Series",
}
