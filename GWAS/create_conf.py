class ConfFile:
    def __init__(self, filename):
        self.filename = filename
        self.data = self._load()

    def _load(self):
        with open(self.filename, 'r') as file:
            content = file.readlines()

        data = {}
        current_section = None
        for line in content:
            line = line.strip()
            if not line or line.startswith('//'):  # Ignore empty lines and comments
                continue
            if line.endswith('{'):
                section_name = line.split('{')[0].strip()
                data[section_name] = {}
                current_section = data[section_name]
            elif line == '}':
                current_section = None
            elif current_section is not None:
                key, value = [x.strip() for x in line.split('=')]
                current_section[key] = value
            else:
                key, value = [x.strip() for x in line.split('=')]
                data[key] = value

        return data

    def get_value(self, section, key=None):
        return self.data[section][key] if key else self.data[section]

    def set_value(self, section, key=None, value=None):
        if isinstance(value, str):
            value = f"\'{value}\'"
        assert value is not None, 'Value cannot be None'
        if key:
            if section in self.data and isinstance(self.data[section], dict):
                self.data[section][key] = value
            else:
                self.data[section] = {key: value}
        else:
            self.data[section] = value
            
    def remove_key(self, section, key):
        if key in self.data[section]:
            del self.data[section][key]

    def save(self, filename):
        with open(filename, 'w') as file:
            for key, value in self.data.items():
                if isinstance(value, dict):
                    file.write(f'{key} {{\n')
                    for k, v in value.items():
                        file.write(f'     {k} = {v}\n')
                    file.write('}\n')
                else:
                    file.write(f'{key} = {value}\n')


if __name__ == "__main__":
    conf = ConfFile("/group/glastonbury/soumick/MyCodes/GitLab/tricorder/GWAS/template.conf")
    # conf.update('params', 'project', 'NewProject')
    # conf.update('params', 'cleanup', False)
    # conf.update('tower', 'accessToken', 'new_token')
    conf.save("/group/glastonbury/soumick/MyCodes/GitLab/tricorder/GWAS/template_new.conf")