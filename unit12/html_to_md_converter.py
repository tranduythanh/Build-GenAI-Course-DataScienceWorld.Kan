#!/usr/bin/env python3
"""
HTML to Markdown Converter
Converts HTML files to clean, well-formatted Markdown.
Specifically designed to handle blog posts and academic content.
"""

import re
import os
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString
from markdownify import markdownify as md
import html2text


class HTMLToMarkdownConverter:
    def __init__(self):
        self.h = html2text.HTML2Text()
        self.h.ignore_links = False
        self.h.ignore_images = False
        self.h.ignore_emphasis = False
        self.h.body_width = 0  # Don't wrap lines
        self.h.unicode_snob = True
        self.h.escape_snob = True
        
    def extract_main_content(self, soup):
        """Extract the main article content, removing navigation and other elements."""
        # Try to find the main content area
        main_content = None
        
        # Look for common content selectors
        selectors = [
            'article.post-single',
            'main article',
            '.post-content',
            'article',
            'main',
            '.content'
        ]
        
        for selector in selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            # Fallback: use body but remove header, nav, footer
            main_content = soup.find('body')
            if main_content:
                # Remove unwanted elements
                for tag in main_content.find_all(['header', 'nav', 'footer', 'script', 'style']):
                    tag.decompose()
        
        return main_content
    
    def clean_html_content(self, soup):
        """Clean and prepare HTML content for conversion."""
        # Remove unwanted elements
        unwanted_tags = [
            'script', 'style', 'nav', 'header', 'footer', 
            'noscript', '.share-buttons', '.post-meta',
            '.breadcrumb', '.pagination', '.toc'
        ]
        
        for tag_name in unwanted_tags:
            for tag in soup.select(tag_name):
                tag.decompose()
        
        # Clean up specific elements
        # Remove empty paragraphs
        for p in soup.find_all('p'):
            if not p.get_text(strip=True):
                p.decompose()
        
        # Convert figure captions to proper format
        for figure in soup.find_all('figure'):
            img = figure.find('img')
            figcaption = figure.find('figcaption')
            if img and figcaption:
                # Create a new structure for better markdown conversion
                img['alt'] = figcaption.get_text(strip=True)
        
        return soup
    
    def post_process_markdown(self, markdown_text):
        """Clean up the converted markdown text."""
        # Fix multiple newlines
        markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
        
        # Fix heading spacing
        markdown_text = re.sub(r'\n(#{1,6})', r'\n\n\1', markdown_text)
        markdown_text = re.sub(r'(#{1,6}.*)\n([^\n#])', r'\1\n\n\2', markdown_text)
        
        # Fix list spacing
        markdown_text = re.sub(r'\n(\*|\d+\.)', r'\n\n\1', markdown_text)
        
        # Fix code block spacing
        markdown_text = re.sub(r'\n(```)', r'\n\n\1', markdown_text)
        markdown_text = re.sub(r'(```)\n([^\n])', r'\1\n\n\2', markdown_text)
        
        # Clean up extra spaces
        markdown_text = re.sub(r'[ \t]+\n', '\n', markdown_text)
        
        # Fix citation format
        markdown_text = re.sub(r'\[(\d+)\]', r'[\1]', markdown_text)
        
        # Remove HTML comments
        markdown_text = re.sub(r'<!--.*?-->', '', markdown_text, flags=re.DOTALL)
        
        # Clean up beginning and end
        markdown_text = markdown_text.strip()
        
        return markdown_text
    
    def convert_file(self, input_file, output_file=None):
        """Convert HTML file to Markdown."""
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Determine output file
        if output_file is None:
            output_file = input_path.with_suffix('.md')
        
        output_path = Path(output_file)
        
        print(f"Converting {input_path} to {output_path}")
        
        # Read HTML content
        with open(input_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract main content
        main_content = self.extract_main_content(soup)
        if not main_content:
            print("Warning: Could not find main content area, using full body")
            main_content = soup
        
        # Clean HTML
        cleaned_soup = self.clean_html_content(main_content)
        
        # Convert to markdown using html2text (better for complex content)
        markdown_content = self.h.handle(str(cleaned_soup))
        
        # Post-process markdown
        final_markdown = self.post_process_markdown(markdown_content)
        
        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_markdown)
        
        print(f"Conversion complete! Output saved to: {output_path}")
        print(f"Original size: {len(html_content):,} characters")
        print(f"Markdown size: {len(final_markdown):,} characters")
        
        return output_path
    
    def convert_string(self, html_string):
        """Convert HTML string to Markdown string."""
        soup = BeautifulSoup(html_string, 'html.parser')
        main_content = self.extract_main_content(soup)
        if not main_content:
            main_content = soup
        
        cleaned_soup = self.clean_html_content(main_content)
        markdown_content = self.h.handle(str(cleaned_soup))
        return self.post_process_markdown(markdown_content)


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert HTML files to Markdown')
    parser.add_argument('input_file', help='Input HTML file path')
    parser.add_argument('-o', '--output', help='Output Markdown file path (optional)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    converter = HTMLToMarkdownConverter()
    
    try:
        output_path = converter.convert_file(args.input_file, args.output)
        if args.verbose:
            print(f"Successfully converted {args.input_file} to {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 