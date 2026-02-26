
    #include <stdio.h>
    #include <math.h>
    #include <stdlib.h>
    #include <string.h>
    #include <time.h>
    #define teta 0.003
    #define NmbEntree 40
    #define NmbCouche 4
    #define Sortie 256
    #define NmbDimensions 164
    #define NmbCaractere 256

    float entree[NmbEntree+1][NmbDimensions];
    int INDICE_ENTREE[NmbEntree+1];

    typedef struct {
        float WQ[NmbDimensions][NmbDimensions];
        float WK[NmbDimensions][NmbDimensions];
        float WV[NmbDimensions][NmbDimensions];

        float Zin[NmbEntree][NmbDimensions]; // SOMME D'UNE NEURONNE : SOMME DE : (POIDS X ENTREE) + BIAIS
        float DELTA_Zout[NmbEntree][NmbDimensions];
        float DELTA_Zin[NmbEntree][NmbDimensions];
        float WZin[NmbDimensions][NmbDimensions];
        float GRADIENT_WZin[NmbDimensions][NmbDimensions];
        float BZin[NmbDimensions];
        float GRADIENT_BZin[NmbDimensions];
        float Hin[NmbEntree][NmbDimensions];
        float DELTA_Hin[NmbEntree][NmbDimensions];
        float Zout[NmbEntree][NmbDimensions]; // SOMME D'UNE NEURONNE : SOMME DE : (POIDS X ENTREE) + BIAIS
        float WZout[NmbDimensions][NmbDimensions];
        float GRADIENT_WZout[NmbDimensions][NmbDimensions];
        float BZout[NmbDimensions];
        float GRADIENT_BZout[NmbDimensions];

        float Q[NmbEntree][NmbDimensions];
        float K[NmbEntree][NmbDimensions];
        float V[NmbEntree][NmbDimensions];
        float DELTA_V[NmbEntree][NmbDimensions];
        float DELTA_K[NmbEntree][NmbDimensions];
        float DELTA_Q[NmbEntree][NmbDimensions];

        float SCORE[NmbEntree][NmbEntree];
        float DELTA_SCORE[NmbEntree][NmbEntree];
        float SOMME_SCORE[NmbEntree];
        float SOFTMAX[NmbEntree][NmbEntree];
        float DELTA_SOFTMAX[NmbEntree][NmbEntree];

        float DELTA_E[NmbEntree][NmbDimensions]; // CORRESPOND AU DELTA DE L'ENTREE
    } COUCHE;

    COUCHE* COUCHE_ORDRE = NULL;

    float W_PROJECTION[NmbDimensions][NmbCaractere];
    float B_PROJECTION[NmbCaractere];
    float PROBABILITE[NmbEntree][NmbCaractere];
    float GRADIENT_W_PROJECTION[NmbDimensions][NmbCaractere];
    float GRADIENT_B_PROJECTION[NmbCaractere];
    float EMBEDDING_TABLE[NmbCaractere][NmbDimensions];
    float GRADIENT_EMBEDDING[NmbCaractere][NmbDimensions];

    float ERREUR[NmbEntree];
    float cible[NmbEntree];
    unsigned char CIBLE_CHAR;
    float POSITION_EMBEDDING[NmbEntree][NmbDimensions];

    unsigned char* BUFFER_TEXTE = NULL;
    long TAILLE_CORPUS = 0;

    float ReLU(float X){
        return (X > 0) ? X : 0.01f * X;
    }
    float ReLU_DERIVEE(float X){
        return (X > 0) ? 1.0f : 0.01f;
    }

    float Limiter(float g) {
        if (g > 1.0f) return 1.0f;
        if (g < -1.0f) return -1.0f;
        return g;
    }
    
    int EstValide() {
        // On vérifie si la probabilité de la dernière prédiction est un nombre valide
        // Si le réseau a explosé, PROBABILITE contiendra des NaN
        for (int caractere = 0; caractere < NmbCaractere; caractere++) {
            if (isnan(PROBABILITE[NmbEntree-1][caractere]) || isinf(PROBABILITE[NmbEntree-1][caractere])) {
                return 0; // Pas valide
            }
        }
        return 1; // Tout va bien
    }

    // PERMET DE CHARGER LE FICHIER DE DONNEES DANS LA RAM (JE NE COMPREND PAS LE CODE, CAR C'EST GEMINI QUI L'A FAIT)
    void ChargerCorpusEnRAM(const char* nomFichier) {
        FILE* f = fopen(nomFichier, "rb");
        if (!f) { printf("Erreur: Impossible de lire le corpus.\n"); exit(1); }
        
        fseek(f, 0, SEEK_END);
        TAILLE_CORPUS = ftell(f);
        rewind(f);

        BUFFER_TEXTE = (unsigned char*)malloc(TAILLE_CORPUS);
        if (!BUFFER_TEXTE) { printf("Erreur: Pas assez de RAM.\n"); exit(1); }

        fread(BUFFER_TEXTE, 1, TAILLE_CORPUS, f);
        fclose(f);
        printf("🚀 Corpus de %ld octets chargé en RAM.\n", TAILLE_CORPUS);
    }

    int InitialisationProchaineLigne() {
    // ON CHOISI UN POINT DE DEPART AU HASARD PARMIS TOUS LE FICHIER
    long position_aleatoire = rand() % (TAILLE_CORPUS - NmbEntree - 1);
    
    // ON REMPLI LES ENTREES
    for (int i = 0; i <= NmbEntree; i++) {
        for(int d=0; d<NmbDimensions; d++){
            unsigned char c = BUFFER_TEXTE[position_aleatoire + i];
            INDICE_ENTREE[i] = (int)(c);
        }
    }
   
    CIBLE_CHAR = BUFFER_TEXTE[position_aleatoire + NmbEntree];

    return 1;
    }

    void InitialisationDeLia(int INDEX_SELECTIONNE){
    for(int d=0; d<NmbDimensions; d++){
        for(int i=0; i<NmbEntree-1; i++){
        entree[i][d] = entree[i+1][d];
        }
        entree[NmbEntree-1][d] = INDEX_SELECTIONNE/255.0f;
    }
    }

    void InitialisationAleatoire(){
        float LIMITE_XAVIER = sqrt(6.0f / (NmbDimensions + NmbDimensions)); // INITIALISATION DE XAVIER

        for(int c=0; c<NmbCouche; c++){
            for(int d=0; d<NmbDimensions; d++){
                for(int g=0; g<NmbDimensions; g++){
                    COUCHE_ORDRE[c].WZout[d][g] = ((float)rand()/RAND_MAX * 2 - 1) * LIMITE_XAVIER;
                    COUCHE_ORDRE[c].WZin[d][g] = ((float)rand()/RAND_MAX * 2 - 1) * LIMITE_XAVIER;
                }
                COUCHE_ORDRE[c].BZin[d] = ((float)rand()/RAND_MAX * 2 - 1) * LIMITE_XAVIER;
                COUCHE_ORDRE[c].BZout[d] = ((float)rand()/RAND_MAX * 2 - 1) * LIMITE_XAVIER;

                for(int g=0; g<NmbDimensions; g++){
                    COUCHE_ORDRE[c].WQ[d][g] = ((float)rand()/RAND_MAX * 2 - 1) * LIMITE_XAVIER;
                    COUCHE_ORDRE[c].WK[d][g] = ((float)rand()/RAND_MAX * 2 - 1) * LIMITE_XAVIER;
                    COUCHE_ORDRE[c].WV[d][g] = ((float)rand()/RAND_MAX * 2 - 1) * LIMITE_XAVIER;
                }
            }
        }

        for(int e=0; e<NmbEntree; e++){
            for(int d=0; d<NmbDimensions; d++){
                POSITION_EMBEDDING[e][d] =
                    ((float)rand()/RAND_MAX * 2 - 1) * 0.01f;
            }
        }

        for(int caractere=0; caractere<NmbCaractere; caractere++){
            B_PROJECTION[caractere] = ((float)rand()/RAND_MAX * 2 - 1) * LIMITE_XAVIER;
            for(int d=0; d<NmbDimensions; d++){
                W_PROJECTION[d][caractere] = ((float)rand()/RAND_MAX * 2 - 1) * LIMITE_XAVIER;
                EMBEDDING_TABLE[caractere][d] = ((float)rand()/RAND_MAX * 2 - 1) * LIMITE_XAVIER;
                GRADIENT_EMBEDDING[caractere][d] = 0;
            }
        }
    }

    void Memoire(){
        FILE* f = fopen("MEMOIRE_IA.bin", "rb");
        if(f == NULL){
            printf("\n\nMemoire introuvable, initialisation aleatoire :\n\n");
            InitialisationAleatoire();
            return;
        }
        for(int c=0; c<NmbCouche; c++){
            for(int d=0; d<NmbDimensions; d++){
                fread(&COUCHE_ORDRE[c].BZin[d], sizeof(float), 1, f);
                fread(&COUCHE_ORDRE[c].BZout[d], sizeof(float), 1, f);
                for(int g=0; g<NmbDimensions; g++){
                    fread(&COUCHE_ORDRE[c].WQ[d][g], sizeof(float), 1, f);
                    fread(&COUCHE_ORDRE[c].WK[d][g], sizeof(float), 1, f);
                    fread(&COUCHE_ORDRE[c].WV[d][g], sizeof(float), 1, f);
                    fread(&COUCHE_ORDRE[c].WZin[d][g], sizeof(float), 1, f);
                    fread(&COUCHE_ORDRE[c].WZout[d][g], sizeof(float), 1, f);
                }
            }
            for(int caractere=0; caractere<NmbCaractere; caractere++){
                fread(&B_PROJECTION[caractere], sizeof(float), 1, f);
                for(int d=0; d<NmbDimensions; d++){
                    fread(&W_PROJECTION[d][caractere], sizeof(float), 1, f);
                }
            }
        }

        fread(EMBEDDING_TABLE, sizeof(float), NmbCaractere * NmbDimensions, f);
        fread(POSITION_EMBEDDING, sizeof(float), NmbEntree * NmbDimensions, f);

        fclose(f);
        
    }

    void Sauvegarde(){
        FILE* f = fopen("MEMOIRE_IA.bin", "wb");
        if(f == NULL){
            printf("\n\nErreur de sauvegarde\n\n");
            return;
        }
        for(int c=0; c<NmbCouche; c++){
            for(int d=0; d<NmbDimensions; d++){
                fwrite(&COUCHE_ORDRE[c].BZin[d], sizeof(float), 1, f);
                fwrite(&COUCHE_ORDRE[c].BZout[d], sizeof(float), 1, f);
                for(int g=0; g<NmbDimensions; g++){
                    fwrite(&COUCHE_ORDRE[c].WQ[d][g], sizeof(float), 1, f);
                    fwrite(&COUCHE_ORDRE[c].WK[d][g], sizeof(float), 1, f);
                    fwrite(&COUCHE_ORDRE[c].WV[d][g], sizeof(float), 1, f);
                    fwrite(&COUCHE_ORDRE[c].WZin[d][g], sizeof(float), 1, f);
                    fwrite(&COUCHE_ORDRE[c].WZout[d][g], sizeof(float), 1, f);
                }
            }
            for(int caractere=0; caractere<NmbCaractere; caractere++){
                fwrite(&B_PROJECTION[caractere], sizeof(float), 1, f);
                for(int d=0; d<NmbDimensions; d++){
                    fwrite(&W_PROJECTION[d][caractere], sizeof(float), 1, f);
                }
            }
        }

        fwrite(EMBEDDING_TABLE, sizeof(float), NmbCaractere * NmbDimensions, f);
        fwrite(POSITION_EMBEDDING, sizeof(float), NmbEntree * NmbDimensions, f);

        fclose(f);
        
    }


//===================================================================================================================================================================


    void MouvementAvant(){

    for(int c=0; c<NmbCouche; c++){
        for(int e=0; e<NmbEntree; e++){
            for(int d=0; d<NmbDimensions; d++){
                COUCHE_ORDRE[c].Zin[e][d] = 0.0f;
                COUCHE_ORDRE[c].Zout[e][d] = 0.0f;
                COUCHE_ORDRE[c].Hin[e][d] = 0.0f;
            }
        }
    }

    for(int e=0; e<NmbEntree; e++){
        int CARACTERE_INDICE = INDICE_ENTREE[e];
        for(int d = 0; d<NmbDimensions; d++){
            entree[e][d] = EMBEDDING_TABLE[CARACTERE_INDICE][d] + POSITION_EMBEDDING[e][d];

        }
    }

    for(int c=0; c<NmbCouche; c++){
        for(int e=0; e<NmbEntree; e++){
            for(int d=0; d<NmbDimensions; d++){
                COUCHE_ORDRE[c].Q[e][d] = 0;
                COUCHE_ORDRE[c].K[e][d] = 0;
                COUCHE_ORDRE[c].V[e][d] = 0;
                for(int g=0; g<NmbDimensions; g++){
                    if(c == 0){
                        COUCHE_ORDRE[c].Q[e][d] += entree[e][g] * COUCHE_ORDRE[c].WQ[g][d];
                        COUCHE_ORDRE[c].K[e][d] += entree[e][g] * COUCHE_ORDRE[c].WK[g][d];
                        COUCHE_ORDRE[c].V[e][d] += entree[e][g] * COUCHE_ORDRE[c].WV[g][d];
                    }
                    else{
                        COUCHE_ORDRE[c].Q[e][d] += COUCHE_ORDRE[c-1].Zout[e][g] * COUCHE_ORDRE[c].WQ[g][d];
                        COUCHE_ORDRE[c].K[e][d] += COUCHE_ORDRE[c-1].Zout[e][g] * COUCHE_ORDRE[c].WK[g][d];
                        COUCHE_ORDRE[c].V[e][d] += COUCHE_ORDRE[c-1].Zout[e][g] * COUCHE_ORDRE[c].WV[g][d];
                    }
                }
            }
        }
    }

    float MAX_SCORE[NmbEntree];
    for(int c=0; c<NmbCouche; c++){
    for(int e=0; e<NmbEntree; e++){
        MAX_SCORE[e] = -INFINITY;
        for(int eplus1=0; eplus1<=e; eplus1++){
            COUCHE_ORDRE[c].SCORE[e][eplus1] = 0;
            for(int d=0; d<NmbDimensions; d++){
                COUCHE_ORDRE[c].SCORE[e][eplus1] += COUCHE_ORDRE[c].Q[e][d] * COUCHE_ORDRE[c].K[eplus1][d];
            }
            COUCHE_ORDRE[c].SCORE[e][eplus1] /= sqrt(NmbDimensions);
            if(COUCHE_ORDRE[c].SCORE[e][eplus1] > MAX_SCORE[e]){
                MAX_SCORE[e] = COUCHE_ORDRE[c].SCORE[e][eplus1]; 
            }
        }
        COUCHE_ORDRE[c].SOMME_SCORE[e] = 0;
        for(int eplus1=0; eplus1<=e; eplus1++){
            COUCHE_ORDRE[c].SOMME_SCORE[e] += expf(COUCHE_ORDRE[c].SCORE[e][eplus1] - MAX_SCORE[e]);
        }
    }
}


    for(int c=0; c<NmbCouche; c++){
        for(int e=0; e<NmbEntree; e++){
            for(int eplus2=0; eplus2<=e; eplus2++){
                COUCHE_ORDRE[c].SOFTMAX[e][eplus2] = 0;
            }
            for(int eplus1=0; eplus1<=e; eplus1++){
                COUCHE_ORDRE[c].SOFTMAX[e][eplus1] = (expf(COUCHE_ORDRE[c].SCORE[e][eplus1] - MAX_SCORE[e]))/COUCHE_ORDRE[c].SOMME_SCORE[e]; // CALCUL DU SOFTMAXij
            }
        }
    }

    for(int c=0; c<NmbCouche; c++){
        for(int e=0; e<NmbEntree; e++){
            for(int d=0; d<NmbDimensions; d++){
                COUCHE_ORDRE[c].Zin[e][d] = 0;
                for(int eplus1=0; eplus1<=e; eplus1++){
                    COUCHE_ORDRE[c].Zin[e][d] += COUCHE_ORDRE[c].SOFTMAX[e][eplus1] * COUCHE_ORDRE[c].V[eplus1][d]; // CALCUL DE Zi,d (Zin)
                }
            }
        }
    }

    for(int c=0; c<NmbCouche; c++){
        for(int e=0; e<NmbEntree; e++){
            for(int d=0; d<NmbDimensions; d++){
                float calcul_ff = COUCHE_ORDRE[c].BZout[d]; 
                
                // Calcul intermédiaire
                for(int g=0; g<NmbDimensions; g++){
                    float hin_temp = COUCHE_ORDRE[c].BZin[d] + (COUCHE_ORDRE[c].Zin[e][g] * COUCHE_ORDRE[c].WZin[g][d]);
                    COUCHE_ORDRE[c].Hin[e][d] = hin_temp; // Stockage pour la backprop
                    calcul_ff += ReLU(hin_temp) * COUCHE_ORDRE[c].WZout[g][d];
                }

                // AJOUT DE LA CONNEXION RÉSIDUELLE (Le "Skip")
                if(c == 0) {
                    COUCHE_ORDRE[c].Zout[e][d] = entree[e][d] + calcul_ff;
                } else {
                    COUCHE_ORDRE[c].Zout[e][d] = COUCHE_ORDRE[c-1].Zout[e][d] + calcul_ff;
                }
            }
        }
    }

    for(int e=0; e<NmbEntree; e++){
        float MAX_SCORE = -INFINITY;
        float scores[NmbCaractere];
        for(int caractere=0; caractere<NmbCaractere; caractere++){
            float SCORE = B_PROJECTION[caractere];
            for(int d=0; d<NmbDimensions; d++){
                SCORE += COUCHE_ORDRE[NmbCouche-1].Zout[e][d] * W_PROJECTION[d][caractere];
            }
            scores[caractere] = SCORE;
            if(SCORE > MAX_SCORE) MAX_SCORE = SCORE;
        }
        float SOMME_EXP = 0;
        for(int caractere=0; caractere<NmbCaractere; caractere++){
            PROBABILITE[e][caractere] = expf(scores[caractere] - MAX_SCORE);
            SOMME_EXP += PROBABILITE[e][caractere];
        }
        for(int caractere=0; caractere<NmbCaractere; caractere++){
            PROBABILITE[e][caractere] /= SOMME_EXP;
        }
    }

    }

    void MouvementArriereRetropropagation(){

    for(int c=NmbCouche-1; c>=0; c--){ // ON REPETE LE PROCESSUS POUR TOUTES LES COUCHES
    if(c == NmbCouche-1){
        int e = NmbEntree - 1; // On ne calcule que pour la dernière position prédite
        
        for(int e=0; e<NmbEntree; e++) {
            for(int d=0; d<NmbDimensions; d++) {
                COUCHE_ORDRE[c].DELTA_Zout[e][d] = 0.0f;
            }
        }
        

        for(int caractere=0; caractere<NmbCaractere; caractere++){
            float erreur_claire = PROBABILITE[e][caractere] - (caractere == CIBLE_CHAR ? 1.0f : 0.0f);
            
            // Accumulation des gradients de la couche de projection
            GRADIENT_B_PROJECTION[caractere] += erreur_claire;
            for(int d=0; d<NmbDimensions; d++){
                GRADIENT_W_PROJECTION[d][caractere] += erreur_claire * COUCHE_ORDRE[c].Zout[e][d];
                
                // 2. Accumulation du delta pour la couche précédente (Somme sur tous les caractères)
                COUCHE_ORDRE[c].DELTA_Zout[e][d] += erreur_claire * W_PROJECTION[d][caractere];
            }
        }
    }

    // ON CALCUL EN PREMIER DELTA_Z_ATTENTION
        for(int g=0; g<NmbDimensions; g++){
            COUCHE_ORDRE[c].GRADIENT_BZin[g]=0;
            for(int d=0; d<NmbDimensions; d++){
                COUCHE_ORDRE[c].GRADIENT_WZout[g][d]=0;
                COUCHE_ORDRE[c].GRADIENT_WZin[g][d]=0;
            }
            COUCHE_ORDRE[c].GRADIENT_BZout[g]=0;
        }

        for(int e=0; e<NmbEntree; e++){ 
            for(int g=0; g<NmbDimensions; g++){
                COUCHE_ORDRE[c].DELTA_Zin[e][g] = 0;
            }
        }

        for (int e = 0; e < NmbEntree; e++) {
        if (c < NmbCouche - 1) {
            for (int d = 0; d < NmbDimensions; d++) {
                COUCHE_ORDRE[c].DELTA_Zout[e][d] = COUCHE_ORDRE[c+1].DELTA_E[e][d];
            }
        }
        for (int g = 0; g < NmbDimensions; g++) {
            float erreur_cumulee = 0.0f;
            for (int d = 0; d < NmbDimensions; d++) {
                erreur_cumulee += COUCHE_ORDRE[c].DELTA_Zout[e][d] * COUCHE_ORDRE[c].WZout[g][d];
            }
            COUCHE_ORDRE[c].DELTA_Hin[e][g] = erreur_cumulee * ReLU_DERIVEE(COUCHE_ORDRE[c].Hin[e][g]);
        }
        for (int g = 0; g < NmbDimensions; g++) {
            COUCHE_ORDRE[c].DELTA_Zin[e][g] = 0.0f;
            for (int d = 0; d < NmbDimensions; d++) {
                COUCHE_ORDRE[c].DELTA_Zin[e][g] += COUCHE_ORDRE[c].DELTA_Hin[e][d] * COUCHE_ORDRE[c].WZin[g][d];
            }
        }
        for (int d = 0; d < NmbDimensions; d++) {
            COUCHE_ORDRE[c].GRADIENT_BZout[d] += COUCHE_ORDRE[c].DELTA_Zout[e][d];
            COUCHE_ORDRE[c].GRADIENT_BZin[d]  += COUCHE_ORDRE[c].DELTA_Hin[e][d];
            
            for (int g = 0; g < NmbDimensions; g++) {
                COUCHE_ORDRE[c].GRADIENT_WZout[g][d] += COUCHE_ORDRE[c].DELTA_Zout[e][d] * ReLU(COUCHE_ORDRE[c].Hin[e][g]);
                COUCHE_ORDRE[c].GRADIENT_WZin[g][d]  += COUCHE_ORDRE[c].DELTA_Hin[e][d] * COUCHE_ORDRE[c].Zin[e][g];
            }
        }
    }

    // ON CALCUL DC/DSOFTMAXij,d(l)

        for(int e=0; e<NmbEntree; e++){
            for(int eplus1=0; eplus1<=e; eplus1++){
                COUCHE_ORDRE[c].DELTA_SOFTMAX[e][eplus1] = 0.0f;
                for(int d=0; d<NmbDimensions; d++){
                        COUCHE_ORDRE[c].DELTA_SOFTMAX[e][eplus1] += COUCHE_ORDRE[c].DELTA_Zin[e][d] * COUCHE_ORDRE[c].V[eplus1][d];
                }
            }
        }

    // ON CALCUL DC/DSCORE

        for(int e=0; e<NmbEntree; e++){
            for(int eplus1=0; eplus1<=e; eplus1++){
            COUCHE_ORDRE[c].DELTA_SCORE[e][eplus1] = 0;
            float SOMME_SOFTMAX=0;
                for(int eplus2=0; eplus2<=e; eplus2++){
                    SOMME_SOFTMAX += (COUCHE_ORDRE[c].DELTA_SOFTMAX[e][eplus2] * COUCHE_ORDRE[c].SOFTMAX[e][eplus2]);
                }
            COUCHE_ORDRE[c].DELTA_SCORE[e][eplus1] = COUCHE_ORDRE[c].SOFTMAX[e][eplus1] * (COUCHE_ORDRE[c].DELTA_SOFTMAX[e][eplus1] - SOMME_SOFTMAX);
            }
        }

    // ON SAIT QUE DC/DVj,d(l) = DC/DZi,d(l) X DZi,d(l)/DVj,d(l) = SOMME DE DELTAi,d X SOFTMAXij
        for(int e=0; e<NmbEntree; e++){
            for(int d=0; d<NmbDimensions; d++){
                COUCHE_ORDRE[c].DELTA_V[e][d] = 0.0f;
            }
        }
        for(int e=0; e<NmbEntree; e++){
            for(int d=0; d<NmbDimensions; d++){
                for(int eplus1=0; eplus1<=e; eplus1++){
                    COUCHE_ORDRE[c].DELTA_V[eplus1][d] += (COUCHE_ORDRE[c].SOFTMAX[e][eplus1] * COUCHE_ORDRE[c].DELTA_Zin[e][d]);
                }
            }
        }

    // ON CHERCHE LE DELTA DE Q  (DC/DQi,d(l))

        for(int e=0; e<NmbEntree; e++){
            for(int d=0; d<NmbDimensions; d++){
                COUCHE_ORDRE[c].DELTA_Q[e][d] = 0;
                for(int eplus1=0; eplus1<=e; eplus1++){
                    COUCHE_ORDRE[c].DELTA_Q[e][d] += COUCHE_ORDRE[c].DELTA_SCORE[e][eplus1] * ((COUCHE_ORDRE[c].K[eplus1][d])/sqrt(NmbDimensions));
                }
            }
        }

    // ON CHERCHE LE DELTA DE K  (DC/DKi,d(l))
        for(int e=0; e<NmbEntree; e++){
            for(int d=0; d<NmbDimensions; d++){
                COUCHE_ORDRE[c].DELTA_K[e][d] = 0;
            }
        }
        for(int e=0; e<NmbEntree; e++){
            for(int eplus1=0; eplus1<=e; eplus1++){
                for(int d=0; d<NmbDimensions; d++){
                    COUCHE_ORDRE[c].DELTA_K[eplus1][d] += COUCHE_ORDRE[c].DELTA_SCORE[e][eplus1] * ((COUCHE_ORDRE[c].Q[e][d])/sqrt(NmbDimensions));
                }
            }
        }

    // ON CHERCHE LE DELTA DE E

        for(int e=0; e<NmbEntree; e++){
            for(int d=0; d<NmbDimensions; d++){
                COUCHE_ORDRE[c].DELTA_E[e][d] = 0;
                for(int g=0; g<NmbDimensions; g++){
                        COUCHE_ORDRE[c].DELTA_E[e][d] += (COUCHE_ORDRE[c].DELTA_Q[e][g] * COUCHE_ORDRE[c].WQ[g][d]) 
                                                      +  (COUCHE_ORDRE[c].DELTA_K[e][g] * COUCHE_ORDRE[c].WK[g][d]) 
                                                      +  (COUCHE_ORDRE[c].DELTA_V[e][g] * COUCHE_ORDRE[c].WV[g][d]);
                }
            }
        }
        
    // ON ACTUALISE LES POIDS : GRADIENTS

        for(int g=0; g<NmbDimensions; g++){
            for(int d=0; d<NmbDimensions; d++){
                float GRADIENT_Q=0;
                float GRADIENT_K=0;
                float GRADIENT_V=0;
                for(int e=0; e<NmbEntree; e++){
                    if(c == 0){
                        GRADIENT_Q += entree[e][g] * COUCHE_ORDRE[c].DELTA_Q[e][d];
                        GRADIENT_K += entree[e][g] * COUCHE_ORDRE[c].DELTA_K[e][d];
                        GRADIENT_V += entree[e][g] * COUCHE_ORDRE[c].DELTA_V[e][d];
                    }
                    else{
                        GRADIENT_Q += COUCHE_ORDRE[c-1].Zout[e][g] * COUCHE_ORDRE[c].DELTA_Q[e][d];
                        GRADIENT_K += COUCHE_ORDRE[c-1].Zout[e][g] * COUCHE_ORDRE[c].DELTA_K[e][d];
                        GRADIENT_V += COUCHE_ORDRE[c-1].Zout[e][g] * COUCHE_ORDRE[c].DELTA_V[e][d];
                    }
                }
                COUCHE_ORDRE[c].WQ[g][d] -= Limiter(GRADIENT_Q) * teta;
                COUCHE_ORDRE[c].WK[g][d] -= Limiter(GRADIENT_K) * teta;
                COUCHE_ORDRE[c].WV[g][d] -= Limiter(GRADIENT_V) * teta;
            }
        }

        for(int caractere=0; caractere<NmbCaractere; caractere++){
            B_PROJECTION[caractere] -= GRADIENT_B_PROJECTION[caractere] * teta;
            GRADIENT_B_PROJECTION[caractere] = 0;
            for(int d=0; d<NmbDimensions; d++){
                W_PROJECTION[d][caractere] -= GRADIENT_W_PROJECTION[d][caractere] * teta;
                GRADIENT_W_PROJECTION[d][caractere] = 0;
            }
        }

        for(int d=0; d<NmbDimensions; d++){
        for(int g=0; g<NmbDimensions; g++){
            COUCHE_ORDRE[c].WZout[d][g] -= teta * COUCHE_ORDRE[c].GRADIENT_WZout[d][g];
            COUCHE_ORDRE[c].WZin[d][g] -= teta * COUCHE_ORDRE[c].GRADIENT_WZin[d][g];
        }
        COUCHE_ORDRE[c].BZout[d] -= teta * COUCHE_ORDRE[c].GRADIENT_BZout[d];
        COUCHE_ORDRE[c].BZin[d] -= teta * COUCHE_ORDRE[c].GRADIENT_BZin[d];
    }

}

    for(int e=0; e<NmbEntree; e++){
        int CARACTERE_INDICE = INDICE_ENTREE[e];
        for(int d=0; d<NmbDimensions; d++){
            EMBEDDING_TABLE[CARACTERE_INDICE][d] -= teta * COUCHE_ORDRE[0].DELTA_E[e][d];
        }
    }

    }

    int main(){
        char CHOIX;
        srand(time(NULL));
        COUCHE_ORDRE = (COUCHE*)malloc(sizeof(COUCHE) * NmbCouche);
        if (COUCHE_ORDRE == NULL) {
            printf("🚨 Erreur critique : Impossible d'allouer la mémoire pour l'IA.\n");
            return 1;
        }
        memset(COUCHE_ORDRE, 0, sizeof(COUCHE) * NmbCouche);

        Memoire();

        // ON CHARGE LES DONNEES D'ENTRAINEMENT
        FILE* fichierEntrainement = fopen("corpus_final.txt", "r"); // OUVERTURE LECTURE
        if(fichierEntrainement == NULL){
            perror("\nErreur lors de l'ouverture du fichier : ");
            return 0;
        }

        ChargerCorpusEnRAM("corpus_final.txt");

        printf("\n\n=====================================================\n\n");
        printf("Fichier ouvert avec succes.\n\n");
        printf("\n\n=====================================================\n\n");

        printf("Que Souhaitez vous faire ? :\n\n't' -> tester l'intelligence artificielle\n'e' -> entrainer l'intelligence artificielle\n\nVotre choix :");
        scanf("%c", &CHOIX);
        if(CHOIX == 't'){ // SI L'UTILISATEUR CHOISI LE TEST
            InitialisationProchaineLigne();
            MouvementAvant();
            int INDEX_SELECTIONNE;
            float MEILLEUR_PROBA = -1.0f;

            printf("\n\n===================================");
            printf("\n\nVOICI LA PHRASE DE DEPART : \n\n");
            // AFFICHER LES ENTREES
            for(int e=0; e<NmbEntree; e++){
                    printf("%c", (unsigned char)INDICE_ENTREE[e]);
            }

            printf("\n\n===================================");
            printf("\n\nVOICI LA SUITE LOGIQUE POUR L'INTELLIGENCE ARTIFICIELLE :\n\n");

            int INDEX_SELECTIONNE_TEST;
            float MEILLEUR_PROBA_TEST = -1.0f;
            float INDEX_ALEATOIRE = 0;
            float SOMME_GLOBALE;
            
                for(int r=0; r<200; r++){
                    MouvementAvant();
                    SOMME_GLOBALE = 0;
                    INDEX_ALEATOIRE = ((float)rand() / (float)RAND_MAX);
                        for(int caractere=0; caractere<NmbCaractere; caractere++){
                            SOMME_GLOBALE += PROBABILITE[NmbEntree-1][caractere];
                            if(INDEX_ALEATOIRE <= SOMME_GLOBALE){
                                INDEX_SELECTIONNE_TEST = caractere;
                                break;
                            }
                        }

                    printf("%c", (char)INDEX_SELECTIONNE_TEST);
                    fflush(stdout);
                    InitialisationDeLia(INDEX_SELECTIONNE_TEST);
                }
        }
        else if(CHOIX == 'e'){// SI L'UTILISATEUR CHOISI L'ENTRAINEMENT
            for(int i=0; i<100000000; i++){
                InitialisationProchaineLigne();
                MouvementAvant();
                MouvementArriereRetropropagation();
                int INDEX_SELECTIONNE_TEST;
                float ERREUR_DU_RESULTAT;
                float MEILLEUR_PROBA_ENTRAINEMENT = -1.0f;
                if(i%200 == 0){
                        float max_proba = -1.0f;
                        INDEX_SELECTIONNE_TEST = 0;
                        for(int car=0; car<NmbCaractere; car++){
                            if(PROBABILITE[NmbEntree-1][car] > max_proba){
                                max_proba = PROBABILITE[NmbEntree-1][car];
                                INDEX_SELECTIONNE_TEST = car;
                            }
                        }
                        int LettreAttendu = CIBLE_CHAR;
                        float ProbaBonneLettre = PROBABILITE[NmbEntree-1][LettreAttendu];
                        ERREUR_DU_RESULTAT = 1.0f - ProbaBonneLettre;
                        printf("Iteration: %d | Cible: '%c' | Predit: '%c' (Confiance: %.2f%%) | Erreur: %.4f\n", 
                                i, (char)CIBLE_CHAR, (char)INDEX_SELECTIONNE_TEST, max_proba*100, ERREUR_DU_RESULTAT);
                        if (EstValide()) {
                            Sauvegarde();
                        } else {
                            printf("ALERTE : NaN détecté à l'itération %d ! Sauvegarde annulée.\n", i);
                            break;
                        }
                    }
                }
            }
        
        

        return(0);
    }
