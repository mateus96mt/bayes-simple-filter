#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 08:30:19 2019

@author: mateus
"""

import numpy as np #biblioteca numérica 
import matplotlib.pyplot as plt #biblioteca gráfica para degrar gráficos
import sympy as sp #biblioteca para manipulação simbólica

#funcao da distribuição normal
def distribuicao_normal(media, desvio_padrao, x):
    mu = media
    sig = desvio_padrao
    
    return (1/(np.sqrt(2*np.pi*(sig**2))))*sp.exp(-(1.0/2.0)*((x - mu)/sig)**2)

#funcao para plotar grafico
def plota_grafico(funcao, simbolo, dominio, num_pontos, cor = "blue"):
    x = simbolo
    domx = dominio
    
    eixo_x = np.linspace(domx[0], domx[1], num_pontos)
    eixo_y = [funcao.subs(x, eixo_x[i]) for i in range(len(eixo_x))]
    
    plt.plot(eixo_x, eixo_y, color=cor)

#funcao com implementacao do filtro de bayes, recebe como parametro as medidas
def filtro_bayes(medidas, posicao_real, desv_pad_ruido):
        
    corplot = ['b', 'g', 'r', 'c', 'm', 'y']
    idxcor = 0
    
    xy = posicao_real
    desv_pad = desv_pad_ruido
    x = sp.Symbol('x')#variavel simbolica 'x'
    
    domx = [xy - 6*desv_pad, xy + 6*desv_pad]#dominio em x para plotar graficos
    
    m = medidas #medidas da latitude do barco
    tam = len(medidas) #numero de medidas
    
    #nao ha conhecimento sobre a posição do barco inicialmente
    #distribuição uniforme
    priori = sp.Mul(1.0/(domx[1] - domx[0]))
    
    #distribuicao normal centrada na primeira medida com desvio padrao dado
    posteriori = distribuicao_normal(m[0], desv_pad, x)
    
    result = 0.0
    for i in range(1, tam+1):
        
        plt.subplots_adjust(wspace=0.4)
        
        plt.subplot(132)
        plota_grafico(priori, x, domx, 100, corplot[idxcor%6])
        plt.title("priori")
        plt.grid()
        
        plt.subplot(131)
        plota_grafico(posteriori, x, domx, 100, corplot[(idxcor+1)%6])
        plt.title("posteriori")
        plt.grid()
        
        #integral de -inf a inf de P(Z|X)*P(X) para normalizar o resultado
        norm = 1.0/(sp.integrate(priori*posteriori, (x, -sp.oo, sp.oo)))
        
        result = sp.Mul(norm, priori*posteriori)
        
        if i<tam:
            posteriori = distribuicao_normal(m[i], desv_pad, x)
            priori = result
            
        plt.subplot(133)
        plt.axvline(xy, color='k', label = "XY")
        plota_grafico(result, x, domx, 100, corplot[(idxcor+2)%6])
        plt.title("Bayes")
        plt.grid()
        idxcor = idxcor + 2
        plt.suptitle("Iteração " + str(i))
        plt.legend(loc="upper left")
        plt.tight_layout()
        fig = plt.gcf()
        fig.set_size_inches(16, 9)
        fig.savefig("it" + str(i) + ".png", dpi=150)
        plt.show();
        plt.clf()
    
    #primeira derivada = 0 nos da o pico onde 'x' é a média
    df = sp.diff(result, x)
    #segunda derivada = 0 nos dá o ponto de inflexão que é o 'x' do desvio padrão
    ddf = sp.diff(df, x)
    
    
    media = float(list(sp.solveset(df, x))[0])

    #a media - o valor de x mais à direita nos da o desvio padrao
#    desvio_padrao = media - list(sp.solveset(ddf, x))[1]
    desvio_padrao = media - float(list(sp.solveset(ddf, x))[0])

    print("FINAL DO PROCESSO DO FILTRO DE BAYES:     MÉDIA: ", media,
          "       DESVIO PADRÃO: ", desvio_padrao)

#funcao main para execucao do codigo
def main():
    #meu numero de matrícula é 102190188, os dois ultimos dígitos são 88
    xy = 88 #posicao real da embarcação, "média" das posições
    
    #medidas de latitude da embarcação segundo o enunciado da questão
    m = np.array([xy + 3.76, xy + 3.04, xy + 6.86, xy + 7.36, xy + 7.30, xy + 4.97])
    
    #desvio padrão do sensor de latitude da embarcação
    desv_pad = 4
    
    filtro_bayes(m, xy, desv_pad)
    
main()