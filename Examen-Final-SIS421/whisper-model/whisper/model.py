#! IMPORTACION DE LAS LIBRERIAS
import base64
import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

#? Importa funciones clave de otros archivos
from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function



#? Verifica si la función scaled_dot_product_attention está disponible en PyTorch
try:
    from torch.nn.functional import scaled_dot_product_attention  # Intenta importar la función
    SDPA_AVAILABLE = True  # Si la importación tiene éxito, marca que está disponible
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None  
    SDPA_AVAILABLE = False  # Marca que la función no está disponible






#! Definimos las dimensiones clave del modelo Whisper
@dataclass
class ModelDimensions:
    #? Define las dimensiones clave del modelo
    
    # Codificador de audio (procesamiento de espectrogramas Mel)
    n_mels: int              #* Número de filtros Mel para la transformacion de audio en el espectrograma
    n_audio_ctx: int         #* Contexto del audio, cantidad de texto a procesar
    n_audio_state: int       #* Estado del codificador de audio, tamaño del vector de estado que el codificador
    n_audio_head: int        #* Número de cabezas de atención en audio
    n_audio_layer: int       #* Número de capas del codificador
    
    # Decodificador de texto (generación de texto)
    n_vocab: int             #* Tamaño del vocabulario, que incluye palabras, tokens de idioma
    n_text_ctx: int          #* Contexto máximo para texto, cantidad máxima de tokens de texto a procesar
    n_text_state: int        #* Tamaño del vector de estado que el decodificador usa para representar el texto durante la generación
    n_text_head: int         #* Número de cabezas de atención en texto
    n_text_layer: int        #* Número de capas del decodificador











#! DEFINIMCIÓN DE CLASES AUXILIARES 
#? Custom Layer Normalization con soporte para diferentes tipos de datos
class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


#? Custom Linear Layer compatible con diferentes tipos de datos
class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

#? Custom Conv1D Layer con manejo de tipos de datos
class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )







#! FUNCIÓN AUXLIAR
#? Genera sinusoides para embeddings posicionales
def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0  # Asegura que el número de canales sea par
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)  # Incremento logarítmico
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))  # Escalas inversas
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]  # Escala temporal
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)  # Combina seno y coseno



#? Context manager para deshabilitar SDPA si es necesario
@contextmanager
def disable_sdpa():
    prev_state = MultiHeadAttention.use_sdpa  # Guarda el estado actual de SDPA
    try:
        MultiHeadAttention.use_sdpa = False  # Deshabilita SDPA temporalmente
        yield  # Permite ejecutar operaciones con SDPA desactivado
    finally:
        MultiHeadAttention.use_sdpa = prev_state  # Restaura el estado original






#! MÓDULOS DEL MODELO WHISPER

#? Implementa la Atención Multi-Cabeza
class MultiHeadAttention(nn.Module):
    use_sdpa = True  # Indica si se debe usar la versión optimizada de atención escalar

    def __init__(self, n_state: int, n_head: int):
        # n_state representa la dimensión de los vectores de entrada y salida
        super().__init__()
        self.n_head = n_head  # Número de cabezas de atención
        
        #* Transformaciones lineales para Q, K, V y salida
        self.query = Linear(n_state, n_state)  # Proyección para el vector Query (Q)
        self.key = Linear(n_state, n_state, bias=False)  # Proyección para el vector Key (K)
        self.value = Linear(n_state, n_state)  # Proyección para el vector Value (V)
        self.out = Linear(n_state, n_state)  # Proyección para combinar las salidas de las cabezas

    #* Forward para cálculos de atención
    def forward(
        self,  
        x: Tensor, # Entrada principal (Q) para la atención
        xa: Optional[Tensor] = None,  # Si es atención cruzada, usa un segundo tensor para K y V
        mask: Optional[Tensor] = None,  # Máscara para posiciones a ignorar, sirve para prevenir acceso a tokens futuros
        kv_cache: Optional[dict] = None,  # Cache para reutilizar cálculos
    ):
        q = self.query(x)  # Calcula el vector Query (Q)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            #* Calcula Key (K) y Value (V) si no están en cache
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            #* Usa el cache si está disponible
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        # Calcula la atención usando Q, K, V y la máscara
        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk  # Aplica la proyección final y retorna la salida de atención


    #? Realiza cálculos de atención (clave para comprensión)
    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape  # Obtiene dimensiones de Batch, Contexto y Estado
        scale = (n_state // self.n_head) ** -0.25  # Escalamiento para estabilizar gradientes
        # Divide Q, K, V en subespacios por cabeza y ajusta las dimensiones
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)


        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
            #* Usa SDPA optimizado si está disponible
            a = scaled_dot_product_attention(
                q, k, v, is_causal=mask is not None and n_ctx > 1
            )
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None  # No calcula QK explícitamente en este modo
        else:
            #* Calcula atención manualmente
            qk = (q * scale) @ (k * scale).transpose(-1, -2)  # Producto punto entre Q y K
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]  # Aplica la máscara si es necesaria
            qk = qk.float()

            # Aplica Softmax y combina con V
            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()  # Desvincula QK del gráfico de gradientes

        return out, qk  # Retorna la salida de atención y las similitudes QK





#? Bloques Residuales con Auto/Atención Cruzada
class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        #* Auto-atención (MultiHeadAttention) con normalización
        self.attn = MultiHeadAttention(n_state, n_head)  # Captura relaciones dentro de la misma secuencia
        self.attn_ln = LayerNorm(n_state)  # Normaliza antes de la auto-atención

        #* Atención cruzada (opcional)
        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None  # Relaciona dos secuencias diferentes
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None  # Normaliza antes de la atención cruzada

        #* Perceptrón multicapa (MLP)
        n_mlp = n_state * 4  # Dimensión intermedia 4x mayor
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)  # Transformaciones lineales y activación
        )
        self.mlp_ln = LayerNorm(n_state)  # Normaliza antes del MLP

    #* Forward aplica bloques residuales y atención
    def forward(
        self,
        x: Tensor,  # Entrada principal
        xa: Optional[Tensor] = None,  # Secuencia secundaria (para atención cruzada)
        mask: Optional[Tensor] = None,  # Máscara opcional para ignorar posiciones
        kv_cache: Optional[dict] = None,  # Cache para reutilizar cálculos
    ):
        #* Auto-atención con conexión residual
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        
        #* Atención cruzada (si está habilitada)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        
        #* Perceptrón multicapa (MLP) con conexión residual
        x = x + self.mlp(self.mlp_ln(x))
        return x  # Salida enriquecida















#? Codificador para procesar audio
class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        #* Capas convolucionales iniciales para extraer características locales
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)

        #* Embeddings posicionales para codificar posiciones temporales
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        #* Bloques residuales con atención multi-cabeza
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )

        #* Normalización final
        self.ln_post = LayerNorm(n_state)

    #? Forward procesa espectrogramas Mel
    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        #* Convoluciones iniciales con activación GELU
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        
        #* Cambia la forma del tensor para que sea compatible con las capas posteriores
        # Entrada: (batch_size, n_state, n_ctx)
        # Salida: (batch_size, n_ctx, n_state)
        x = x.permute(0, 2, 1)

        #* Verifica que el tamaño coincida con los embeddings posicionales
        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"

        #* Suma embeddings posicionales al tensor procesado
        x = (x + self.positional_embedding).to(x.dtype)

        #* Pasa por los bloques residuales
        for block in self.blocks:
            x = block(x)

        #* Normalización final
        x = self.ln_post(x)
        return x




#? Decodificador para generar texto
class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        #* Embedding de tokens de texto
        self.token_embedding = nn.Embedding(n_vocab, n_state) 
        #* Embedding posicional aprendido
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state)) 

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)  #* Pila de bloques residuales con atención cruzada
            ]
        )
        
        self.ln = LayerNorm(n_state) #* Normalización al final


        #? Máscara de atención para prevenir el acceso a tokens futuros
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)


    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]  #* Embedding posicional ajustado
        )
        
        x = x.to(xa.dtype)  #? Ajuste de tipo de datos para coincidencia con audio

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)  #* Procesamiento con bloques residuales

        x = self.ln(x)  #* Normalización al final
        
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float() #* Predicción de tokens (logits)

        return logits #* Retorna logits para tokens de texto    




#todo CLASE PRINCIPAL DEL MODELO WHISPER
class Whisper(nn.Module):
    
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims  #? Dimensiones del modelo definidas previamente
        
        #* Inicialización del codificador
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        ) 
        #* Inicialización del decodificador
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        ) 
        
        #* Define cabezas de alineación temporal por defecto
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        
        
        all_heads[self.dims.n_text_layer // 2 :] = True # * Solo las cabezas de la mitad de las capas hacia adelante son verdaderas
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)


    def set_alignment_heads(self, dump: bytes):
        
        #* cabezas específicas para alineación temporal
        array = np.frombuffer(gzip.decompress(base64.b85decode(dump)), dtype=bool).copy()
        
        mask = torch.from_numpy(array).reshape(self.dims.n_text_layer, self.dims.n_text_head)
        
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    #? Metodos funcionales 
    def embed_audio(self, mel: torch.Tensor):
        #* Extrae características del audio usando el codificador
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        #* Genera logits (probabilidades) para los tokens
        return self.decoder(tokens, audio_features)

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        #* Combina codificación de audio y decodificación de texto
        return self.decoder(tokens, self.encoder(mel))


    #? PROPIEDADES 
    @property
    def device(self):
        #* Devuelve el dispositivo (CPU o GPU) donde está el modelo
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        #* Indica si el modelo es multilingüe
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        #* Calcula el número de idiomas soportados
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    #? Métodos avanzados
    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        
        """
        Habilita hooks para guardar claves y valores intermedios
        usados en la atención para mejorar eficiencia.
        """
        
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    #? Funcionalidades adicionales importadas de otros archivos
    detect_language = detect_language_function #* Detección de idioma
    transcribe = transcribe_function  #* Transcripción de audio a texto
    decode = decode_function   #* Decodificación completa de texto
