# Propuesta de aplicación para control de patrimonio

## 1. Objetivo

Crear una aplicación web que permita gestionar el patrimonio familiar y empresarial de forma centralizada, reemplazando la dependencia de múltiples hojas de Excel y permitiendo:

- ver el patrimonio consolidado por propietario,
- separar activos por categoría y tipo,
- registrar compras y ventas de acciones,
- manejar cuentas bancarias compartidas,
- aplicar distintas reglas de valoración según el tipo de activo,
- mantener un historial y reportes claros.

La propuesta está pensada para que pueda desplegarse en un servidor y usarse de forma diaria, sin perder la estructura que ya existe en los archivos Excel actuales.

---

## 2. Contexto actual

Actualmente existen tres archivos Excel:

- Patrimonio ARS
- Patrimonio Patri
- Patrimonio Comun

Cada uno tiene una estructura similar:

- una hoja resumen,
- varias hojas con los valores por concepto,
- información agrupada por categorías y activos.

Esto indica que ya existe una lógica de negocio útil, pero está distribuida en archivos estáticos y no permite:

- cálculos automáticos más complejos,
- trazabilidad de movimientos,
- reglas de valoración personalizadas,
- control de cuentas compartidas,
- consolidación por propietario.

---

## 3. Propuesta general

Diseñar una aplicación de gestión patrimonial con las siguientes capacidades:

1. Un sistema central de datos para todos los activos.
2. Un motor de valoración configurable por tipo de activo.
3. Un modelo de propiedad flexible para personas y entidades.
4. Importación desde los Excel actuales para no perder la información histórica.
5. Una interfaz web simple y clara para registrar, consultar y reportar.

---

## 4. Requisitos funcionales

### 4.1 Gestión de activos

La aplicación debe permitir registrar activos de distintos tipos, por ejemplo:

- Casa
- Inmueble
- Hereditas
- Acciones
- Cuentas bancarias
- Efectivo
- Empresas o participaciones

Cada activo debe tener:

- nombre,
- categoría,
- tipo,
- valoración actual,
- propietario o propietarios,
- fecha de alta,
- observaciones.

### 4.2 Tipos de valoración

La app debe permitir seleccionar una regla de valoración distinta por activo o por tipo de activo.

Se recomienda implementar al menos estos modelos:

- Valoración por mercado menos deuda
  - útil para casa e inmueble
  - fórmula: valor de mercado - deuda pendiente

- Valoración por activo neto de empresa
  - útil para Hereditas o sociedades
  - fórmula: activos corrientes + activos fijos valorados - pasivos

- Valoración por valor de mercado directo
  - útil para acciones o instrumentos financieros

- Valoración personalizada por tipo
  - para futuros casos como otra sociedad o estructura distinta

Este punto es clave porque usted ya mencionó que el mismo tipo de activo puede requerir una lógica distinta según el caso.

### 4.3 Propiedad compartida

La app debe soportar que un activo o cuenta pertenezca a varios propietarios.

Ejemplos:

- cuenta bancaria compartida entre Antonio y Patri,
- activo propiedad del 50% para cada uno,
- activo propiedad de una compañía y de personas,
- participaciones en una sociedad con varios titulares.

El sistema debe permitir definir:

- propietarios,
- porcentaje de participación,
- tipo de relación (persona, pareja, empresa, sociedad).

Para cuentas compartidas, el valor del patrimonio para cada propietario se calculará proporcionalmente al porcentaje asignado.

### 4.4 Acciones y movimientos

Debe poder registrarse:

- compra de acciones,
- venta de acciones,
- transferencias entre cuentas,
- aportes o retiros,
- cambios de valor de activos.

Esto permitirá que el patrimonio se actualice con el tiempo y no dependa solo de una captura manual puntual.

### 4.5 Reportes y dashboard

La app debe mostrar al menos:

- patrimonio total,
- patrimonio por propietario,
- patrimonio por categoría,
- patrimonio por tipo de activo,
- evolución histórica,
- cuentas bancarias y saldos,
- exposición a acciones y empresas,
- resumen por entidad.

---

## 5. Modelo de datos propuesto

### 5.1 Entidades principales

- Personas o entidades
  - Antonio
  - Patri
  - Empresa nueva
  - otras sociedades

- Activos
  - casa
  - inmueble
  - acciones
  - cuentas bancarias
  - empresa
  - etc.

- Cuentas
  - cuentas bancarias
  - cuentas de inversión
  - cuentas de caja

- Movimientos
  - compras
  - ventas
  - transferencias
  - aportes
  - retiros

- Reglas de valoración
  - nombre
  - tipo
  - fórmula o lógica

- Participaciones
  - activo -> propietario -> porcentaje

### 5.2 Estructura recomendada

Se recomienda separar la información en estas capas:

- catálogo de activos,
- catálogo de propietarios,
- movimientos históricos,
- valoración actual,
- snapshots de patrimonio por fecha.

Esto permitirá no solo ver el estado actual, sino también consultar cómo ha cambiado el patrimonio con el tiempo.

---

## 6. Diseño funcional sugerido

### 6.1 Pantallas principales

- Dashboard general
  - resumen del patrimonio total
  - distribución por propietario
  - evolución mensual o anual

- Gestión de activos
  - alta, edición y baja de activos
  - asignación de propietarios
  - selección de regla de valoración

- Gestión de cuentas
  - cuentas bancarias y de inversión
  - saldos actuales
  - cuentas compartidas

- Movimientos
  - registrar compras/ventas/transfers
  - ver historial cronológico

- Reportes
  - balance patrimonial
  - detalle por activo
  - detalle por cuenta
  - comparación de periodos

---

## 7. Arquitectura técnica recomendada

### 7.1 Backend

Se recomienda un backend en Python con:

- FastAPI
- SQLAlchemy
- PostgreSQL
- Alembic para migraciones

Esto ofrece una base sólida, mantenimiento sencillo y buena escalabilidad.

### 7.2 Frontend

Para la interfaz web, se recomienda:

- React + Vite, o
- Next.js

Para un primer despliegue, React con Vite suele ser más directo y ligero.

### 7.3 Importación desde Excel

La app debe poder importar los archivos actuales para crear una primera carga de datos.

Se recomienda:

- leer los archivos Excel existentes,
- mapear las hojas a tablas internas,
- cargar los datos como una migración inicial,
- conservar la posibilidad de exportar de nuevo a Excel si se desea.

### 7.4 Despliegue

Para un servidor, la mejor opción es usar:

- Docker Compose
- PostgreSQL en contenedor
- app web en contenedor
- nginx o reverse proxy opcional

Esto hace el despliegue más simple y repetible.

---

## 8. Plan de implementación

### Fase 1 - MVP

Objetivo: tener una versión funcional para operar el patrimonio diario.

Incluye:

- registro básico de activos,
- registro de cuentas,
- asignación de propietarios,
- cálculo simple de valor,
- importación inicial desde Excel,
- dashboard básico.

### Fase 2 - Movimientos y valoración avanzada

Incluye:

- compra/venta de acciones,
- transferencias entre cuentas,
- reglas de valoración más complejas,
- soporte para cuentas compartidas,
- consolidado por propietario.

### Fase 3 - Reportes y trazabilidad

Incluye:

- histórico de patrimonio,
- snapshots por fecha,
- reportes por categoría y entidad,
- exportación a Excel o PDF.

### Fase 4 - Escalado y refinamiento

Incluye:

- usuarios y roles,
- auditoría de cambios,
- notificaciones,
- integración con bancos o brokers,
- mejores reportes y paneles.

---

## 9. Decisiones recomendadas

### Opción recomendada

Construir una aplicación web moderna con backend en Python y frontend en React, con PostgreSQL como base de datos y soporte para importar los Excel actuales.

### Razón

Porque permite:

- escalar en el futuro,
- separar la lógica de negocio de los archivos estáticos,
- manejar reglas complejas de propiedad y valoración,
- publicar la app en un servidor de forma estable.

---

## 10. Supuestos tomados para esta propuesta

Para que la propuesta sea accionable, he asumido lo siguiente:

- la información actual en Excel puede usarse como fuente de datos inicial,
- los activos se pueden agrupar por categoría y tipo,
- los propietarios pueden ser personas o entidades,
- las cuentas compartidas se resuelven por porcentaje de participación,
- el nuevo company puede incorporarse como una entidad adicional con su propia lógica de valoración,
- el objetivo inicial es cubrir el caso práctico de patrimonio personal y familiar, no una plataforma financiera completa.

---

## 11. Recomendación final

La mejor solución no es seguir trabajando con hojas de cálculo como fuente principal, sino convertir ese modelo en una aplicación que:

- centralice la información,
- automatice los cálculos,
- permita manejar propietarios múltiples,
- soporte distintos métodos de valoración,
- se pueda desplegar en un servidor y usar de forma continua.

Mi recomendación es comenzar con un MVP enfocado en:

1. activos,
2. cuentas,
3. propietarios,
4. movimientos,
5. valoración configurable,
6. dashboard inicial.

Con esa base, la app podrá crecer sin volver a reescribir todo el modelo.
