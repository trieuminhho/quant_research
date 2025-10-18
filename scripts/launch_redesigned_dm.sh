#!/bin/bash
# Launch script for redesigned Data Management page

echo "🚀 Launching QuantSearch Data Management (Redesigned)..."
echo ""
echo "📊 Features:"
echo "  ✅ Interactive KPI cards that filter the table"
echo "  ✅ Advanced filtering with search and sort"
echo "  ✅ Bulk selection controls (Select All, Deselect, Missing, Invert)"
echo "  ✅ Floating actions bar for batch operations"
echo "  ✅ Modern design system with smooth animations"
echo ""
echo "🌐 Opening browser at http://localhost:8501"
echo ""

source venv/bin/activate
streamlit run ui/pages/data_management.py
