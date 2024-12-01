import streamlit as st
from typing import Dict, Optional, List, Any
import json
import os
import yaml
from googletrans import Translator
import logging

class LanguageManager:
    def __init__(self, translations_dir: str = "translations/"):
        self.translations_dir = translations_dir
        self.logger = logging.getLogger(__name__)
        self.translator = Translator()
        self.available_languages = self._load_available_languages()
        self._translation_cache = {}  # Add cache
        self._init_translations()
        
    def _init_translations(self):
        """Initialize translation files"""
        os.makedirs(self.translations_dir, exist_ok=True)
        
        # Create default English translations if not exists
        en_file = os.path.join(self.translations_dir, "en.yaml")
        if not os.path.exists(en_file):
            default_translations = {
                "general": {
                    "app_title": "Data Analysis Dashboard",
                    "welcome": "Welcome to the dashboard",
                    "loading": "Loading...",
                    "error": "An error occurred",
                    "success": "Operation successful"
                },
                "navigation": {
                    "home": "Home",
                    "upload": "Upload Data",
                    "analyze": "Analyze",
                    "visualize": "Visualize",
                    "settings": "Settings"
                },
                "analysis": {
                    "start": "Start Analysis",
                    "processing": "Processing data...",
                    "results": "Analysis Results",
                    "export": "Export Results"
                },
                "visualization": {
                    "chart_types": "Chart Types",
                    "bar_chart": "Bar Chart",
                    "line_chart": "Line Chart",
                    "scatter_plot": "Scatter Plot",
                    "customize": "Customize Chart"
                },
                "data": {
                    "upload": "Upload File",
                    "preview": "Data Preview",
                    "columns": "Columns",
                    "rows": "Rows",
                    "missing": "Missing Values"
                }
            }
            
            with open(en_file, 'w', encoding='utf-8') as f:
                yaml.dump(default_translations, f, allow_unicode=True)

    def _load_available_languages(self) -> Dict[str, str]:
        """Load list of available languages"""
        return {
            'en': 'English',
            'es': 'Español',
            'fr': 'Français',
            'de': 'Deutsch',
            'ja': '日本語',
            'zh': '中文',
            'ko': '한국어',
            'th': 'ไทย',
            'vi': 'Tiếng Việt',
            'ru': 'Русский'
        }

    def get_translation(self, key: str, lang: str = 'en') -> str:
        """Get translation for a key in specified language"""
        try:
            lang_file = os.path.join(self.translations_dir, f"{lang}.yaml")
            
            # Create language file if it doesn't exist
            if not os.path.exists(lang_file):
                self._create_language_file(lang)
            
            with open(lang_file, 'r', encoding='utf-8') as f:
                translations = yaml.safe_load(f)
                
            # Handle nested keys (e.g., "general.welcome")
            keys = key.split('.')
            value = translations
            for k in keys:
                value = value[k]
            
            return value
            
        except Exception as e:
            self.logger.error(f"Error getting translation: {str(e)}")
            return key

    def _create_language_file(self, lang: str):
        """Create new language file by translating English content"""
        try:
            # Load English translations
            with open(os.path.join(self.translations_dir, "en.yaml"), 'r', encoding='utf-8') as f:
                en_translations = yaml.safe_load(f)
            
            # Translate all values
            translated = self._translate_dict(en_translations, 'en', lang)
            
            # Save translated content
            lang_file = os.path.join(self.translations_dir, f"{lang}.yaml")
            with open(lang_file, 'w', encoding='utf-8') as f:
                yaml.dump(translated, f, allow_unicode=True)
                
        except Exception as e:
            self.logger.error(f"Error creating language file: {str(e)}")

    def _load_translations(self, lang: str) -> Dict:
        """Load translations for a specific language with error handling"""
        try:
            lang_file = os.path.join(self.translations_dir, f"{lang}.yaml")
            if os.path.exists(lang_file):
                with open(lang_file, 'r', encoding='utf-8') as f:
                    translations = yaml.safe_load(f)
                    return translations if translations else {}
            return {}
        except Exception as e:
            self.logger.error(f"Error loading translations for {lang}: {str(e)}")
            return {}

    def _translate_dict(self, d: Dict, src: str, dest: str) -> Dict:
        """Recursively translate dictionary values"""
        translated = {}
        for key, value in d.items():
            if isinstance(value, dict):
                translated[key] = self._translate_dict(value, src, dest)
            else:
                try:
                    translation = self.translator.translate(
                        value, src=src, dest=dest
                    )
                    translated[key] = translation.text
                except Exception as e:
                    self.logger.error(f"Translation error: {str(e)}")
                    translated[key] = value
        return translated

    def show_language_interface(self):
        """Show language management interface in Streamlit"""
        st.subheader("🌐 Language Settings")
        
        # Language selection
        selected_lang = st.selectbox(
            "Select Language",
            options=list(self.available_languages.keys()),
            format_func=lambda x: self.available_languages[x],
            key="selected_language"
        )
        
        if selected_lang != st.session_state.get('language'):
            st.session_state.language = selected_lang
            st.rerun()
        
        # Translation management
        st.subheader("Translation Management")
        
        operation = st.radio(
            "Select Operation",
            ["View Translations", "Add Translation", "Update Translation"]
        )
        
        if operation == "View Translations":
            lang_file = os.path.join(self.translations_dir, f"{selected_lang}.yaml")
            if os.path.exists(lang_file):
                with open(lang_file, 'r', encoding='utf-8') as f:
                    translations = yaml.safe_load(f)
                st.json(translations)
            else:
                st.warning(f"No translations available for {self.available_languages[selected_lang]}")
                
        elif operation == "Add Translation":
            new_key = st.text_input("Translation Key (e.g., general.new_key)")
            new_text = st.text_input("Translation Text")
            
            if new_key and new_text and st.button("Add Translation"):
                try:
                    lang_file = os.path.join(self.translations_dir, f"{selected_lang}.yaml")
                    
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        translations = yaml.safe_load(f)
                    
                    # Handle nested keys
                    keys = new_key.split('.')
                    current = translations
                    for key in keys[:-1]:
                        if key not in current:
                            current[key] = {}
                        current = current[key]
                    current[keys[-1]] = new_text
                    
                    with open(lang_file, 'w', encoding='utf-8') as f:
                        yaml.dump(translations, f, allow_unicode=True)
                    
                    st.success("Translation added successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error adding translation: {str(e)}")
                    
        elif operation == "Update Translation":
            lang_file = os.path.join(self.translations_dir, f"{selected_lang}.yaml")
            if os.path.exists(lang_file):
                with open(lang_file, 'r', encoding='utf-8') as f:
                    translations = yaml.safe_load(f)
                
                # Flatten translations for easier selection
                flat_translations = self._flatten_dict(translations)
                
                key_to_update = st.selectbox(
                    "Select key to update",
                    options=list(flat_translations.keys())
                )
                
                if key_to_update:
                    current_value = flat_translations[key_to_update]
                    new_value = st.text_input("New Translation", value=current_value)
                    
                    if st.button("Update Translation"):
                        try:
                            # Update nested dictionary
                            keys = key_to_update.split('.')
                            current = translations
                            for key in keys[:-1]:
                                current = current[key]
                            current[keys[-1]] = new_value
                            
                            with open(lang_file, 'w', encoding='utf-8') as f:
                                yaml.dump(translations, f, allow_unicode=True)
                            
                            st.success("Translation updated successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error updating translation: {str(e)}")

            else:
                st.warning(f"No translations available for {self.available_languages[selected_lang]}")

    def _flatten_dict(self, d: Dict, parent_key: str = '') -> Dict:
        """Flatten nested dictionary with dot notation"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def auto_translate(self, text: str, target_lang: str) -> str:
        """Automatically translate text to target language"""
        try:
            translation = self.translator.translate(text, dest=target_lang)
            return translation.text
        except Exception as e:
            self.logger.error(f"Auto translation error: {str(e)}")
            return text

    def bulk_translate(self, source_lang: str, target_lang: str) -> bool:
        """Bulk translate from source language to target language"""
        try:
            source_file = os.path.join(self.translations_dir, f"{source_lang}.yaml")
            with open(source_file, 'r', encoding='utf-8') as f:
                source_translations = yaml.safe_load(f)
            
            translated = self._translate_dict(
                source_translations,
                source_lang,
                target_lang
            )
            
            target_file = os.path.join(self.translations_dir, f"{target_lang}.yaml")
            with open(target_file, 'w', encoding='utf-8') as f:
                yaml.dump(translated, f, allow_unicode=True)
            
            return True
        except Exception as e:
            self.logger.error(f"Bulk translation error: {str(e)}")
            return False

    def validate_translations(self) -> Dict[str, List[str]]:
        """Validate translations across all languages"""
        missing_keys = {}
        english_keys = set(self._flatten_dict(self._load_translations('en')).keys())
        
        for lang in self.available_languages.keys():
            if lang != 'en':
                lang_keys = set(self._flatten_dict(self._load_translations(lang)).keys())
                missing = english_keys - lang_keys
                if missing:
                    missing_keys[lang] = list(missing)
        
        return missing_keys

    def _load_translations(self, lang: str) -> Dict:
        """Load translations for a specific language"""
        lang_file = os.path.join(self.translations_dir, f"{lang}.yaml")
        if os.path.exists(lang_file):
            with open(lang_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    def export_translations(self, format: str = 'json') -> str:
        """Export all translations to specified format"""
        all_translations = {}
        for lang in self.available_languages.keys():
            all_translations[lang] = self._load_translations(lang)
        
        if format == 'json':
            return json.dumps(all_translations, ensure_ascii=False, indent=2)
        elif format == 'yaml':
            return yaml.dump(all_translations, allow_unicode=True)
        return ''

    def import_translations(self, content: str, format: str = 'json') -> bool:
        """Import translations from specified format"""
        try:
            if format == 'json':
                translations = json.loads(content)
            elif format == 'yaml':
                translations = yaml.safe_load(content)
            else:
                return False
            
            for lang, trans in translations.items():
                lang_file = os.path.join(self.translations_dir, f"{lang}.yaml")
                with open(lang_file, 'w', encoding='utf-8') as f:
                    yaml.dump(trans, f, allow_unicode=True)
            
            return True
        except Exception as e:
            self.logger.error(f"Import error: {str(e)}")
            return False

    def show_bulk_translation_interface(self):
        """Show bulk translation interface"""
        st.subheader("Bulk Translation")
        
        col1, col2 = st.columns(2)
        with col1:
            source_lang = st.selectbox(
                "Source Language",
                options=list(self.available_languages.keys()),
                format_func=lambda x: self.available_languages[x],
                key="source_lang"
            )
        
        with col2:
            target_lang = st.selectbox(
                "Target Language",
                options=[l for l in self.available_languages.keys() if l != source_lang],
                format_func=lambda x: self.available_languages[x],
                key="target_lang"
            )
        
        if st.button("Start Bulk Translation"):
            with st.spinner("Translating..."):
                if self.bulk_translate(source_lang, target_lang):
                    st.success("Bulk translation completed!")
                else:
                    st.error("Error during bulk translation")

    def show_validation_interface(self):
        """Show translation validation interface"""
        st.subheader("Translation Validation")
        
        missing_keys = self.validate_translations()
        
        if missing_keys:
            st.warning("Missing translations found!")
            for lang, keys in missing_keys.items():
                with st.expander(f"Missing in {self.available_languages[lang]}"):
                    st.write(keys)
                    
                    if st.button(f"Auto-translate missing keys for {lang}"):
                        try:
                            en_translations = self._load_translations('en')
                            lang_translations = self._load_translations(lang)
                            
                            for key in keys:
                                en_value = self._get_nested_value(en_translations, key)
                                if en_value:
                                    translated = self.auto_translate(en_value, lang)
                                    self._set_nested_value(lang_translations, key, translated)
                            
                            lang_file = os.path.join(self.translations_dir, f"{lang}.yaml")
                            with open(lang_file, 'w', encoding='utf-8') as f:
                                yaml.dump(lang_translations, f, allow_unicode=True)
                                
                            st.success(f"Auto-translation completed for {lang}!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error during auto-translation: {str(e)}")
        else:
            st.success("All translations are complete!")

    def _get_nested_value(self, d: Dict, key: str):
        """Get value from nested dictionary using dot notation"""
        keys = key.split('.')
        value = d
        for k in keys:
            value = value.get(k)
            if value is None:
                return None
        return value

    def _set_nested_value(self, d: Dict, key: str, value: Any):
        """Set value in nested dictionary using dot notation"""
        keys = key.split('.')
        current = d
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value

    def clear_cache(self):
        """Clear the translation cache"""
        self._translation_cache = {}

    def refresh_translations(self):
        """Refresh all translation files"""
        self.clear_cache()
        for lang in self.available_languages.keys():
            if lang != 'en':
                self._create_language_file(lang)
        return True
    
    def validate_translation_key(self, key: str) -> bool:
        """Validate if a translation key follows the correct format"""
        if not key or not isinstance(key, str):
            return False
        parts = key.split('.')
        return all(part.isalnum() or '_' in part for part in parts)