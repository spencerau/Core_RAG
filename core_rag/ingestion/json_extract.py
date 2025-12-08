from typing import Dict, List


class JSONContentExtractor:
    def __init__(self, config: dict):
        self.config = config
    
    def extract_content_for_embedding(self, json_data, section_path: str = "") -> List[Dict]:
        if isinstance(json_data, list):
            return self._extract_array_content(json_data)
        elif isinstance(json_data, dict):
            if 'program' in json_data and 'sections' in json_data:
                return self._extract_structured_content(json_data)
            return self._extract_dict_content(json_data)
        return []
    
    def _extract_array_content(self, json_array: List) -> List[Dict]:
        content_items = []
        for item in json_array:
            if not isinstance(item, dict):
                continue
            if 'title' in item and 'url' in item:
                text = f"Title: {item.get('title', '')}\nURL: {item.get('url', '')}\n"
                text += f"Category: {item.get('category', '')}\nDescription: {item.get('description', '')}"
                content_items.append({
                    'text': text.strip(), 'section_type': 'resource_link',
                    'section_name': item.get('category', 'General Resource'),
                    'section_classification': 'Resource Links'
                })
            else:
                text_parts = [f"{k.title()}: {v}" for k, v in item.items() if isinstance(v, (str, int, float, bool))]
                if text_parts:
                    content_items.append({
                        'text': '\n'.join(text_parts), 'section_type': 'structured_data',
                        'section_name': 'Data Entry', 'section_classification': 'Structured Information'
                    })
        return content_items
    
    def _extract_dict_content(self, json_dict: Dict) -> List[Dict]:
        text_parts = []
        for key, value in json_dict.items():
            if isinstance(value, (str, int, float, bool)):
                text_parts.append(f"{key.title()}: {value}")
            elif isinstance(value, list) and value and isinstance(value[0], str):
                text_parts.append(f"{key.title()}: {', '.join(value)}")
        if text_parts:
            return [{'text': '\n'.join(text_parts), 'section_type': 'general_info',
                     'section_name': 'General Information', 'section_classification': 'Information'}]
        return []
    
    def _extract_structured_content(self, json_data: Dict) -> List[Dict]:
        content_items = []
        header_text = ""
        for key in self.config.get('ingestion', {}).get('json_top_level_keys', []):
            if key in json_data:
                header_text += f"{key.replace('_', ' ').title()}: {json_data[key]}\n"
        
        if 'requirements' in json_data:
            req = json_data['requirements']
            header_text += "\nRequirements:\n"
            if isinstance(req, dict):
                for rk, rv in req.items():
                    if isinstance(rv, dict):
                        for sk, sv in rv.items():
                            header_text += f"- {sk.replace('_', ' ').title()}: {sv}\n"
                    else:
                        header_text += f"- {rk.replace('_', ' ').title()}: {rv}\n"
        
        if 'total_credits' in json_data:
            header_text += f"\nTotal Credits: {json_data['total_credits']}\n"
        
        content_items.append({'text': header_text.strip(), 'section_type': 'overview', 'section_name': 'Overview'})
        
        for section in json_data.get('sections', []):
            content_items.extend(self._process_section(section))
        
        if 'sections' in json_data:
            summary = f"Program Structure Summary for {json_data.get('program', 'Unknown Program')}:\n\n"
            summary += "This program consists of the following requirement categories:\n"
            for s in json_data['sections']:
                summary += f"- {s.get('name', 'Unknown')} ({s.get('credits', 'N/A')} credits)\n"
            if 'total_credits' in json_data:
                summary += f"\nTotal Program Credits: {json_data['total_credits']}"
            content_items.append({'text': summary.strip(), 'section_type': 'program_structure',
                                  'section_name': 'Program Structure Summary'})
        return content_items
    
    def _process_section(self, section: Dict) -> List[Dict]:
        name = section.get('name', 'Unknown Section')
        text = f"Section: {name}\nCredits: {section.get('credits', 'N/A')}\n"
        if 'notes' in section:
            text += f"Notes: {section['notes']}\n"
        
        if section.get('courses'):
            text += "\nCourses:\n"
            for c in section['courses']:
                text += f"- {c.get('course_number', '')}: {c.get('name', '')}\n"
                if c.get('prerequisite'):
                    text += f"  Prerequisite: {c['prerequisite']}\n"
                if c.get('description'):
                    text += f"  Description: {c['description']}\n"
        
        for seq_key, seq_label in [('math_sequences', 'Math Sequence'), ('approved_sequences', 'Approved Sequence')]:
            if seq_key in section:
                text += f"\n{seq_label} Options:\n"
                for i, seq in enumerate(section[seq_key], 1):
                    text += f"{seq_label} {i}:\n"
                    for c in seq.get('courses', []):
                        text += f"  - {c.get('course_number', '')}: {c.get('name', '')}\n"
                        if c.get('description'):
                            text += f"    {c['description']}\n"
        
        return [{'text': text.strip(), 'section_type': 'section',
                 'section_name': name, 'section_classification': name}]
