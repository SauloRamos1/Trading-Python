import openpyxl

# link para acesso ao arquivo xlsx
# http://www.b3.com.br/pt_br/market-data-e-indices/indices/indices-amplos/indice-ibovespa-ibovespa-composicao-da-carteira.htm
# Tabela considera as variações na participação de cada um dos papéis na composição total do índice, apuradas para a abertura do dia.


data = openpyxl.load_workbook('2020.02.19 Ibovespa.xlsx')
sheet = data.active

for row in sheet:
    sheet.append(row)

for row in sheet.iter_rows(min_row=2, min_col=1, max_row=74, max_col=1): # da linha 2 até 74 numéro total no arquivo xlsx
    for cell in row:
        acao = (cell.value + ".SA")
        print(acao)
    print()





