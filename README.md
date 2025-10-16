# Materials Database (Streamlit + Supabase)

Plataforma de caracterização de materiais com módulos automáticos de otimização:
- Espectroscopia Raman (RamanChada2)
- Resistividade (4 Pontas)
- Tensiometria (Ângulo de contato)

# Como rodar no Streamlit Cloud / GitHub

1. Suba o repositório no GitHub.
2. Configure o app no [Streamlit Cloud](https://share.streamlit.io).
3. Em **Settings → Secrets**, copie o conteúdo de `.streamlit/secrets.toml.example` e insira suas chaves Supabase reais.
4. Execute o app:
```
streamlit run app.py
```

# Estrutura dos módulos

- raman_processing.py → análise espectral avançada com RamanChada2.
- resistivity_processing.py → cálculo de resistência, resistividade e condutividade.
- contact_angle_processing.py → análise de molhabilidade e energia superficial.

## Referências científicas
- [RamanChada2 (H2020 CHARISMA)](https://github.com/h2020charisma/ramanchada2)
- [J. Raman Spectrosc. 2023, 54(5)](https://doi.org/10.1002/jrs.6789)
- [Appl. Spectrosc. 2022, 76(8)](https://doi.org/10.1177/00037028221090988)
